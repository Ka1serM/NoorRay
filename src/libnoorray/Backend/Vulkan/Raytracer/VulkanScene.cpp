#include "VulkanScene.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <span>
#include <unordered_map>
#include <vector>

#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Objects/GaussianInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Scene.h"

namespace
{
static_assert(sizeof(Vertex) == 52,
    "Raytracer shader packed vertex ABI expects 52-byte Vertex records");

gpu::Buffer<std::byte> upload_bytes(gpu::Device& device, const void* data, const std::size_t size)
{
    if (size == 0)
        return {};
    auto buffer = device.buffer<std::byte>(size);
    device.upload(buffer, std::span<const std::byte>(
        static_cast<const std::byte*>(data), size));
    return buffer;
}

struct ProxyMesh { std::vector<glm::vec3> vertices; std::vector<uint32_t> indices; };

float proxyInradius(const ProxyMesh& mesh)
{
    float result = std::numeric_limits<float>::max();
    for (size_t face = 0; face < mesh.indices.size(); face += 3u)
    {
        const glm::vec3 a = mesh.vertices[mesh.indices[face]];
        const glm::vec3 b = mesh.vertices[mesh.indices[face + 1u]];
        const glm::vec3 c = mesh.vertices[mesh.indices[face + 2u]];
        const glm::vec3 normal = glm::normalize(glm::cross(b - a, c - a));
        result = std::min(result, std::abs(glm::dot(normal, a)));
    }
    if (!(result > 0.0f) || !std::isfinite(result))
        throw std::runtime_error("Gaussian proxy has an invalid inradius");
    return result;
}

ProxyMesh gaussianProxyMesh(const GaussianProxyType type)
{
    if (type == GaussianProxyType::Octahedron) {
        return {{{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}},
            {0,2,4, 0,5,2, 0,3,5, 0,4,3, 1,4,2, 1,2,5, 1,5,3, 1,3,4}};
    }
    const float phi = 0.5f * (1.0f + std::sqrt(5.0f));
    ProxyMesh mesh{{{-1,phi,0},{1,phi,0},{-1,-phi,0},{1,-phi,0},
        {0,-1,phi},{0,1,phi},{0,-1,-phi},{0,1,-phi},
        {phi,0,-1},{phi,0,1},{-phi,0,-1},{-phi,0,1}},
        {0,11,5, 0,5,1, 0,1,7, 0,7,10, 0,10,11,
         1,5,9, 5,11,4, 11,10,2, 10,7,6, 7,1,8,
         3,9,4, 3,4,2, 3,2,6, 3,6,8, 3,8,9,
         4,9,5, 2,4,11, 6,2,10, 8,6,7, 9,8,1}};
    for (glm::vec3& vertex : mesh.vertices)
        vertex = glm::normalize(vertex);
    const uint32_t subdivisions = type == GaussianProxyType::IcosphereLevel2
        ? 2u : type == GaussianProxyType::Icosphere ? 1u : 0u;
    for (uint32_t level = 0; level < subdivisions; ++level) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
        auto midpoint = [&](uint32_t a, uint32_t b) {
            if (a > b) std::swap(a, b);
            const auto key = std::pair{a, b};
            if (const auto found = midpoints.find(key); found != midpoints.end())
                return found->second;
            const uint32_t index = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(glm::normalize(mesh.vertices[a] + mesh.vertices[b]));
            midpoints.emplace(key, index);
            return index;
        };
        std::vector<uint32_t> refined;
        refined.reserve(mesh.indices.size() * 4u);
        for (size_t face = 0; face < mesh.indices.size(); face += 3u) {
            const uint32_t a = mesh.indices[face], b = mesh.indices[face + 1u],
                c = mesh.indices[face + 2u];
            const uint32_t ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
            refined.insert(refined.end(), {a,ab,ca, b,bc,ab, c,ca,bc, ab,bc,ca});
        }
        mesh.indices = std::move(refined);
    }
    return mesh;
}

}

VulkanScene::VulkanScene(gpu::Device& gpu_device, const Scene& scene)
    : gpuDevice(gpu_device)
{
    build(scene);
}

void VulkanScene::build(const Scene& scene)
{
    std::unordered_map<const MeshAsset*, uint32_t> meshIndices;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMeshAsset())
            continue;
        const MeshAsset* asset = &instance->getMeshAsset();
        if (!meshIndices.contains(asset))
        {
            const uint32_t index = static_cast<uint32_t>(meshes.size());
            meshIndices.emplace(asset, index);
            buildMesh(*asset);
        }
    }
    buildGaussians(scene);
    buildTopLevel(scene);
    buildSceneData(scene);
}

void VulkanScene::buildGaussians(const Scene& scene)
{
    std::vector<Gaussian> records;
    std::vector<float> opacities;
    std::vector<uint16_t> shCoefficients;
    std::vector<uint32_t> instanceOffsets;
    const uint32_t coefficientCount = sphericalHarmonicsCoefficientCount(
        scene.getRenderSettings().gaussianRenderSphericalHarmonics);
    gaussianShCoefficientCount_ = coefficientCount;

    for (const auto& instance : scene.getGaussianInstances())
    {
        if (!instance || !instance->hasGaussianAsset())
            continue;
        instanceOffsets.push_back(static_cast<uint32_t>(records.size()));
        const auto& source = instance->getGaussianAsset().getHostGaussians();
        const glm::mat4 world = instance->getWorldTransform().getMatrix();
        const glm::mat3 linear(world);
        const size_t recordBegin = records.size();
        records.insert(records.end(), source.begin(), source.end());
        // GaussianAsset stores R*S and its center in object space.  The
        // acceleration path historically applied the instance transform via
        // the proxy TLAS transform; Vulkan's compact Gaussian arrays are
        // intentionally flat, so bake the same affine transform into each
        // immutable record at upload time.  This preserves anisotropy while
        // avoiding a second per-Gaussian instance buffer in the shader.
        for (size_t recordIndex = recordBegin; recordIndex < records.size(); ++recordIndex)
        {
            Gaussian& gaussian = records[recordIndex];
            for (uint32_t column = 0; column < 3; ++column)
                gaussian.transform[column] = linear * gaussian.transform[column];
            gaussian.transform[3] = glm::vec3(
                world * glm::vec4(gaussian.transform[3], 1.0f));
        }
        for (const Gaussian& gaussian : source)
        {
            opacities.push_back(gaussian.opacity);
            const uint32_t available = std::min(
                gaussian.sphericalHarmonics.count, coefficientCount);
            const size_t oldSize = shCoefficients.size();
            shCoefficients.resize(oldSize + coefficientCount * 3u, uint16_t{0});
            for (uint32_t coefficient = 0; coefficient < available; ++coefficient)
            {
                std::memcpy(shCoefficients.data() + oldSize + coefficient * 3u,
                    gaussian.sphericalHarmonics.values.data() + coefficient * 3u,
                    sizeof(uint16_t) * 3u);
            }
        }

        // Use one shared tiny proxy BLAS for every Gaussian instance
        // for every Gaussian. Expanding every transformed proxy into unique
        // vertices turns a 1.6M-splat level-2 scene into 533M triangles.
        gaussianTransforms.reserve(gaussianTransforms.size() + source.size());
        for (std::size_t local = 0; local < source.size(); ++local)
        {
            const Gaussian& gaussian = records[recordBegin + local];
            gpu::float4x4 transform{};
            for (uint32_t row = 0; row < 3; ++row)
            {
                for (uint32_t column = 0; column < 3; ++column)
                    transform.values[row][column] = gaussian.transform[column][row];
                transform.values[row][3] = gaussian.transform[3][row];
            }
            transform.values[3][3] = 1.0f;
            gaussianTransforms.push_back(transform);
        }
    }
    gaussianCount_ = static_cast<uint32_t>(records.size());
    if (records.empty())
    {
        gaussianShCoefficientCount_ = 0;
        return;
    }

    const ProxyMesh unitProxy = gaussianProxyMesh(
        scene.getRenderSettings().gaussianProxyType);
    gaussianProxyType_ = static_cast<uint32_t>(scene.getRenderSettings().gaussianProxyType);
    gaussianCutoffSigma_ = scene.getRenderSettings().gaussianCutoffSigma;
    const float proxyScale = scene.getRenderSettings().gaussianCutoffSigma
        / proxyInradius(unitProxy);
    gaussianProxyTriangleCount_ = static_cast<uint32_t>(unitProxy.indices.size() / 3u);
    std::vector<gpu::float3> positions;
    positions.reserve(unitProxy.vertices.size());
    for (const glm::vec3 vertex : unitProxy.vertices)
        positions.push_back({vertex.x * proxyScale, vertex.y * proxyScale,
            vertex.z * proxyScale});
    gaussianProxy.positions = gpuDevice.buffer<gpu::float3>(positions.size());
    gaussianProxy.indices = gpuDevice.buffer<std::uint32_t>(unitProxy.indices.size());
    gpuDevice.upload(gaussianProxy.positions, std::span<const gpu::float3>(positions));
    gpuDevice.upload(gaussianProxy.indices,
        std::span<const std::uint32_t>(unitProxy.indices));
    const gpu::TriangleGeometry proxyGeometry{gaussianProxy.positions.ptr(),
        gaussianProxy.indices.ptr(), gaussianProxyTriangleCount_, false};
    gaussianProxy.blas = gpuDevice.build_blas(
        std::span<const gpu::TriangleGeometry>(&proxyGeometry, 1));

    gaussianRecords_ = upload_bytes(gpuDevice, records.data(),
        records.size() * sizeof(Gaussian));
    gaussianOpacities_ = upload_bytes(gpuDevice, opacities.data(),
        opacities.size() * sizeof(float));
    gaussianShCoefficients_ = upload_bytes(gpuDevice, shCoefficients.data(),
        shCoefficients.size() * sizeof(uint16_t));
    gaussianInstanceOffsets_ = upload_bytes(gpuDevice, instanceOffsets.data(),
        instanceOffsets.size() * sizeof(uint32_t));
}

namespace
{
uint32_t alignOffset(const uint32_t offset, const uint32_t alignment)
{
    return (offset + alignment - 1u) & ~(alignment - 1u);
}

uint32_t appendBytes(std::vector<uint8_t>& bytes, const void* data,
    const size_t size, const uint32_t alignment = 16u)
{
    const uint32_t offset = alignOffset(static_cast<uint32_t>(bytes.size()), alignment);
    bytes.resize(offset + size);
    if (size != 0 && data != nullptr)
        std::memcpy(bytes.data() + offset, data, size);
    return offset;
}
}

void VulkanScene::buildSceneData(const Scene& scene)
{
    std::vector<const MeshAsset*> assets;
    std::unordered_map<const MeshAsset*, uint32_t> meshIndices;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMeshAsset())
            continue;
        const MeshAsset* asset = &instance->getMeshAsset();
        if (!meshIndices.contains(asset))
        {
            meshIndices.emplace(asset, static_cast<uint32_t>(assets.size()));
            assets.push_back(asset);
        }
    }

    std::vector<VulkanSceneMeshRecord> meshRecords(assets.size());
    std::vector<VulkanSceneInstanceRecord> instanceRecords;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMeshAsset())
            continue;
        VulkanSceneInstanceRecord record{};
        const glm::mat4 transform = instance->getWorldTransform().getMatrix();
        // Slang is compiled row-major. Store the conventional row-major
        // affine matrix while GLM remains column-major on the host.
        for (uint32_t row = 0; row < 3; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                record.objectToWorld[row * 4u + column] = transform[column][row];
        record.meshIndex = meshIndices.at(&instance->getMeshAsset());
        instanceRecords.push_back(record);
    }

    VulkanSceneHeader header{};
    header.meshCount = static_cast<uint32_t>(meshRecords.size());
    header.instanceCount = static_cast<uint32_t>(instanceRecords.size());
    std::vector<uint8_t> bytes(sizeof(VulkanSceneHeader), 0);
    header.meshOffset = appendBytes(bytes, meshRecords.data(),
        meshRecords.size() * sizeof(VulkanSceneMeshRecord));
    header.instanceOffset = appendBytes(bytes, instanceRecords.data(),
        instanceRecords.size() * sizeof(VulkanSceneInstanceRecord));
    sceneInstanceOffset_ = header.instanceOffset;

    for (size_t meshIndex = 0; meshIndex < assets.size(); ++meshIndex)
    {
        const MeshAsset& asset = *assets[meshIndex];
        const auto& vertices = asset.getHostVertices();
        const auto& sourceIndices = asset.getHostIndices();
        std::vector<uint32_t> generatedIndices;
        if (sourceIndices.empty()) {
            generatedIndices.resize(vertices.size());
            for (uint32_t i = 0; i < generatedIndices.size(); ++i)
                generatedIndices[i] = i;
        }
        const auto& indices = sourceIndices.empty() ? generatedIndices : sourceIndices;
        const auto& faces = asset.getHostFaces();
        VulkanSceneMeshRecord& record = meshRecords[meshIndex];
        record.vertexOffset = appendBytes(bytes, vertices.data(),
            vertices.size() * sizeof(Vertex));
        record.indexOffset = appendBytes(bytes, indices.data(),
            indices.size() * sizeof(uint32_t));
        record.faceOffset = appendBytes(bytes, faces.data(),
            faces.size() * sizeof(Face));
        record.vertexCount = static_cast<uint32_t>(vertices.size());
        record.indexCount = static_cast<uint32_t>(indices.size());
        record.faceCount = static_cast<uint32_t>(faces.size());
        const auto& materialIds = asset.getMaterialIds();
        record.materialOffset = appendBytes(bytes, materialIds.data(),
            materialIds.size() * sizeof(uint32_t));
        record.materialCount = static_cast<uint32_t>(materialIds.size());
        if (header.vertexDataOffset == 0)
            header.vertexDataOffset = record.vertexOffset;
        if (header.indexDataOffset == 0)
            header.indexDataOffset = record.indexOffset;
        if (header.faceDataOffset == 0)
            header.faceDataOffset = record.faceOffset;
    }

    header.totalBytes = static_cast<uint32_t>(bytes.size());
    std::memcpy(bytes.data(), &header, sizeof(header));
    // Mesh records were written before their offsets were known, so publish
    // the final records after all geometry chunks have been appended.
    std::memcpy(bytes.data() + header.meshOffset, meshRecords.data(),
        meshRecords.size() * sizeof(VulkanSceneMeshRecord));
    sceneData_ = upload_bytes(gpuDevice, bytes.data(), bytes.size());
}

uint32_t VulkanScene::buildMesh(const MeshAsset& asset)
{
    const auto& sourceVertices = asset.getHostVertices();
    const auto& sourceIndices = asset.getHostIndices();
    if (sourceVertices.empty() || sourceVertices.size() > 0x00FFFFFFu)
        throw std::runtime_error("Vulkan scene mesh has no vertices or too many vertices: "
            + asset.getName());

    std::vector<uint32_t> indices = sourceIndices;
    if (indices.empty())
    {
        if (sourceVertices.size() % 3 != 0)
            throw std::runtime_error("Vulkan scene mesh has non-triangular unindexed geometry: "
                + asset.getName());
        indices.resize(sourceVertices.size());
        for (uint32_t i = 0; i < indices.size(); ++i)
            indices[i] = i;
    }
    if (indices.size() % 3 != 0)
        throw std::runtime_error("Vulkan scene mesh index count is not divisible by three: "
            + asset.getName());
    for (const uint32_t index : indices)
        if (index >= sourceVertices.size())
            throw std::runtime_error("Vulkan scene mesh index exceeds vertex count: "
                + asset.getName());

    std::vector<gpu::float3> positions;
    positions.reserve(sourceVertices.size());
    for (const auto& vertex : sourceVertices)
        positions.push_back({vertex.position.x, vertex.position.y, vertex.position.z});

    MeshState state{
        gpuDevice.buffer<gpu::float3>(positions.size()),
        gpuDevice.buffer<std::uint32_t>(indices.size()),
        {}};
    gpuDevice.upload(state.positions, std::span<const gpu::float3>(positions));
    gpuDevice.upload(state.indices, std::span<const std::uint32_t>(indices));
    const gpu::TriangleGeometry geometry{
        state.positions.ptr(), state.indices.ptr(),
        static_cast<std::uint32_t>(indices.size() / 3), false};
    state.blas = gpuDevice.build_blas(std::span<const gpu::TriangleGeometry>(&geometry, 1));
    meshes.push_back(std::move(state));
    return static_cast<uint32_t>(meshes.size() - 1);
}

void VulkanScene::buildTopLevel(const Scene& scene)
{
    std::unordered_map<const MeshAsset*, uint32_t> meshIndices;
    tlasInstances.clear();
    meshInstanceAssetIndices.clear();
    uint32_t meshInstanceIndex = 0;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMeshAsset())
            continue;
        const MeshAsset* asset = &instance->getMeshAsset();
        auto found = meshIndices.find(asset);
        if (found == meshIndices.end())
        {
            // The build order is the same as the first pass. Reconstructing
            // this map from the scene keeps the records compact without
            // exposing mutable renderer state through Scene.
            const uint32_t index = static_cast<uint32_t>(meshIndices.size());
            found = meshIndices.emplace(asset, index).first;
        }
        tlasInstances.push_back({meshes[found->second].blas,
            instance->getWorldTransform().getGpuTransform(),
            meshInstanceIndex++, 0u, 0xff});
        meshInstanceAssetIndices.push_back(found->second);
    }
    // instanceCustomIndex is 24 bits, so it cannot address more than 16.7M
    // splats. The hit shaders instead recover the ID from the traversal-derived
    // InstanceIndex(), which is not stored in the instance record and therefore
    // carries no bit-width limit. This mirrors the OptiX backend, which read
    // optixGetInstanceIndex() and left OptixInstance::instanceId at zero.
    for (uint32_t gaussianId = 0; gaussianId < gaussianTransforms.size(); ++gaussianId)
    {
        tlasInstances.push_back({gaussianProxy.blas, gaussianTransforms[gaussianId],
            0u, 2u, 0xff});
    }
    meshInstanceCount_ = meshInstanceIndex;
    gaussianInstanceCount_ = static_cast<uint32_t>(scene.getGaussianInstances().size());
    instanceCount_ = static_cast<uint32_t>(tlasInstances.size());
    if (tlasInstances.empty())
        return;
    tlas_ = gpuDevice.build_tlas(std::span<const gpu::Instance>(tlasInstances));
}

bool VulkanScene::updateMutableData(const Scene& scene, const bool updateGaussians)
{
    uint32_t currentMeshInstances = 0;
    for (const auto& instance : scene.getMeshInstances())
        currentMeshInstances += instance && instance->hasMeshAsset() ? 1u : 0u;
    const auto& gaussianInstances = scene.getGaussianInstances();
    uint64_t currentGaussianCount = 0;
    for (const auto& instance : gaussianInstances)
        if (instance && instance->hasGaussianAsset())
            currentGaussianCount += instance->getGaussianAsset().getHostGaussians().size();

    const uint32_t coefficientCount = sphericalHarmonicsCoefficientCount(
        scene.getRenderSettings().gaussianRenderSphericalHarmonics);
    if (currentMeshInstances != meshInstanceCount_
        || gaussianInstances.size() != gaussianInstanceCount_
        || currentGaussianCount != gaussianCount_
        || (gaussianCount_ != 0
            && (coefficientCount != gaussianShCoefficientCount_
                || static_cast<uint32_t>(scene.getRenderSettings().gaussianProxyType)
                    != gaussianProxyType_
                || scene.getRenderSettings().gaussianCutoffSigma != gaussianCutoffSigma_)))
        return false;

    std::vector<VulkanSceneInstanceRecord> meshRecords;
    meshRecords.reserve(meshInstanceCount_);
    uint32_t meshIndex = 0;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMeshAsset())
            continue;
        VulkanSceneInstanceRecord record{};
        const glm::mat4 transform = instance->getWorldTransform().getMatrix();
        for (uint32_t row = 0; row < 3; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                record.objectToWorld[row * 4u + column] = transform[column][row];
        record.meshIndex = meshInstanceAssetIndices[meshIndex];
        meshRecords.push_back(record);
        tlasInstances[meshIndex].transform = instance->getWorldTransform().getGpuTransform();
        ++meshIndex;
    }
    if (!meshRecords.empty())
    {
        const auto bytes = std::as_bytes(std::span<const VulkanSceneInstanceRecord>(meshRecords));
        gpuDevice.upload(sceneData_, bytes, sceneInstanceOffset_);
    }

    if (gaussianCount_ != 0 && updateGaussians)
    {
        std::vector<Gaussian> records;
        std::vector<float> opacities;
        std::vector<uint16_t> shCoefficients;
        std::vector<uint32_t> offsets;
        records.reserve(gaussianCount_);
        opacities.reserve(gaussianCount_);
        shCoefficients.reserve(static_cast<size_t>(gaussianCount_)
            * coefficientCount * SphericalHarmonicsChannelCount);
        offsets.reserve(gaussianInstances.size());
        gaussianTransforms.clear();
        gaussianTransforms.reserve(gaussianCount_);

        for (const auto& instance : gaussianInstances)
        {
            if (!instance || !instance->hasGaussianAsset())
                return false;
            offsets.push_back(static_cast<uint32_t>(records.size()));
            const auto& source = instance->getGaussianAsset().getHostGaussians();
            const glm::mat4 world = instance->getWorldTransform().getMatrix();
            const glm::mat3 linear(world);
            for (const Gaussian& sourceGaussian : source)
            {
                Gaussian gaussian = sourceGaussian;
                for (uint32_t column = 0; column < 3; ++column)
                    gaussian.transform[column] = linear * gaussian.transform[column];
                gaussian.transform[3] = glm::vec3(
                    world * glm::vec4(gaussian.transform[3], 1.0f));
                records.push_back(gaussian);
                opacities.push_back(sourceGaussian.opacity);
                const uint32_t available = std::min(
                    sourceGaussian.sphericalHarmonics.count, coefficientCount);
                const size_t oldSize = shCoefficients.size();
                shCoefficients.resize(oldSize + coefficientCount * 3u, uint16_t{0});
                for (uint32_t coefficient = 0; coefficient < available; ++coefficient)
                    std::memcpy(shCoefficients.data() + oldSize + coefficient * 3u,
                        sourceGaussian.sphericalHarmonics.values.data() + coefficient * 3u,
                        sizeof(uint16_t) * 3u);

                gpu::float4x4 proxyTransform{};
                for (uint32_t row = 0; row < 3; ++row)
                {
                    for (uint32_t column = 0; column < 3; ++column)
                        proxyTransform.values[row][column] = gaussian.transform[column][row];
                    proxyTransform.values[row][3] = gaussian.transform[3][row];
                }
                proxyTransform.values[3][3] = 1.0f;
                gaussianTransforms.push_back(proxyTransform);
            }
        }

        gpuDevice.upload(gaussianRecords_, std::as_bytes(std::span<const Gaussian>(records)));
        gpuDevice.upload(gaussianOpacities_, std::as_bytes(std::span<const float>(opacities)));
        if (!shCoefficients.empty())
            gpuDevice.upload(gaussianShCoefficients_,
                std::as_bytes(std::span<const uint16_t>(shCoefficients)));
        gpuDevice.upload(gaussianInstanceOffsets_,
            std::as_bytes(std::span<const uint32_t>(offsets)));
        for (uint32_t gaussian = 0; gaussian < gaussianCount_; ++gaussian)
            tlasInstances[meshInstanceCount_ + gaussian].transform = gaussianTransforms[gaussian];
    }

    if (tlas_)
        gpuDevice.update_tlas(tlas_, std::span<const gpu::Instance>(tlasInstances));
    return true;
}
