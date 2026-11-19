#include "NoorRay/Gpu/MeshBuffers.h"

#include <unordered_set>

#include "NoorRay/Gpu/BufferUpload.h"

namespace NoorRay::Gpu {

namespace {

std::uint64_t getKey(const Scene::MeshHandle handle) {
    return std::uint64_t{handle.generation} << 32 | handle.index;
}

template <class T>
std::size_t getBytes(const NoorRHI::Buffer<T>& buffer) {
    return buffer ? buffer.getByteSize() : 0;
}

bool isAlphaTested(const std::uint32_t material, const std::span<const std::uint32_t> materialFlags) {
    return material < materialFlags.size() && (materialFlags[material] & Shaders::materialFlagAlphaTest) != 0;
}

//! A BLAS without any-hit skips alpha tests, so a mesh whose instance overrides it with an alpha-tested material is not opaque.
std::vector<std::uint8_t> findAlphaTestedOverrides(const Scene::Scene& scene, const std::span<const std::uint32_t> materialFlags) {
    std::vector<std::uint8_t> meshes(scene.getMeshes().size(), 0);
    for (const Scene::Instance& instance : scene.getInstances().getValues()) {
        const std::uint32_t mesh = scene.getMeshes().denseIndex(instance.mesh);
        if (mesh < meshes.size() && isAlphaTested(scene.getMaterials().denseIndex(instance.material), materialFlags))
            meshes[mesh] = 1;
    }
    return meshes;
}

template <class T>
NoorRHI::Ptr<T> getPointer(const NoorRHI::Buffer<T>& buffer, const bool used) {
    return used && buffer ? buffer.getPtr() : NoorRHI::Ptr<T>{};
}

}

std::expected<void, Core::Error> MeshBuffers::upload(NoorRHI::Device& device, const Scene::Mesh& mesh, MeshGpu& gpu) {
    if (auto result = uploadValues<NoorRHI::float3>(device, gpu.positions, mesh.positions, false); !result)
        return result;
    if (auto result = uploadValues<std::uint32_t>(device, gpu.indices, mesh.indices, false); !result)
        return result;
    if (auto result = uploadValues<std::uint32_t>(device, gpu.normals, mesh.normals, false); !result)
        return result;
    if (auto result = uploadValues<std::uint32_t>(device, gpu.tangents, mesh.tangents, false); !result)
        return result;
    for (std::size_t set = 0; set < Scene::maxUvSets; ++set)
        if (!mesh.uvs[set].empty())
            if (auto result = uploadValues<std::uint32_t>(device, gpu.uvs[set], mesh.uvs[set], false); !result)
                return result;
    if (!mesh.colors.empty())
        if (auto result = uploadValues<std::uint32_t>(device, gpu.colors, mesh.colors, false); !result)
            return result;
    gpu.blasStale = true;
    return {};
}

std::expected<void, Core::Error> MeshBuffers::uploadMaterials(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::Mesh& mesh,
                                                              const std::span<const std::uint32_t> materialFlags, const bool alphaTestedOverride,
                                                              MeshGpu& gpu) {
    std::vector<std::uint32_t> triangleMaterials(mesh.getTriangleCount(), 0);
    bool opaque = !alphaTestedOverride;
    for (const Scene::MeshSubset& subset : mesh.subsets) {
        const std::uint32_t material = scene.getMaterials().denseIndex(subset.material);
        const std::uint32_t value = material < materialFlags.size() ? material : 0;
        if (isAlphaTested(material, materialFlags))
            opaque = false;
        const std::uint32_t first = subset.firstIndex / 3;
        const std::uint32_t last = std::min<std::uint32_t>(first + subset.indexCount / 3, mesh.getTriangleCount());
        for (std::uint32_t triangle = first; triangle < last; ++triangle)
            triangleMaterials[triangle] = value;
    }
    if (opaque != gpu.opaque)
        gpu.blasStale = true;
    gpu.opaque = opaque;
    return uploadValues<std::uint32_t>(device, gpu.triangleMaterials, triangleMaterials, false);
}

std::expected<void, Core::Error> MeshBuffers::update(NoorRHI::Device& device, const Scene::Scene& scene, const Scene::SceneChange& change,
                                                     const std::span<const std::uint32_t> materialFlags) {
    const Core::SlotMap<Scene::Mesh>& sceneMeshes = scene.getMeshes();
    const bool topology = change.has(Scene::Change::MeshTopology) || meshes.size() != sceneMeshes.size();
    const bool materials = topology || change.has(Scene::Change::MaterialSet);
    if (!topology && !materials && !change.has(Scene::Change::MeshPositions))
        return {};
    const std::vector<std::uint8_t> alphaTestedOverrides = findAlphaTestedOverrides(scene, materialFlags);
    std::unordered_set<std::uint64_t> live;
    dense.assign(sceneMeshes.size(), nullptr);
    for (std::uint32_t index = 0; index < sceneMeshes.size(); ++index) {
        const Scene::Mesh& mesh = sceneMeshes.getValues()[index];
        const std::uint64_t key = getKey(sceneMeshes.keyAt(index));
        live.insert(key);
        std::unique_ptr<MeshGpu>& gpu = meshes[key];
        const bool created = !gpu;
        if (created)
            gpu = std::make_unique<MeshGpu>();
        dense[index] = gpu.get();
        if (created)
            if (auto result = upload(device, mesh, *gpu); !result)
                return result;
        if (created || materials)
            if (auto result = uploadMaterials(device, scene, mesh, materialFlags, alphaTestedOverrides[index] != 0, *gpu); !result)
                return result;
    }
    std::erase_if(meshes, [&](const auto& entry) { return !live.contains(entry.first); });
    const Scene::SceneChange::Range& dirty = change.getMeshRange();
    if (!dirty.isEmpty())
        for (std::uint32_t index = dirty.begin; index < std::min<std::uint32_t>(dirty.end, getCount()); ++index) {
            if (auto result = Core::check(dense[index]->positions.upload(sceneMeshes.getValues()[index].positions)); !result)
                return result;
            dense[index]->blasRefit = true;
        }
    std::vector<Shaders::MeshRecord> values(dense.size());
    byteSize = 0;
    for (std::uint32_t index = 0; index < dense.size(); ++index) {
        const Scene::Mesh& mesh = sceneMeshes.getValues()[index];
        const MeshGpu& gpu = *dense[index];
        Shaders::MeshRecord& record = values[index];
        const bool bounded = mesh.bounds.min.x <= mesh.bounds.max.x;
        record.boundsMin = bounded ? Core::float4{mesh.bounds.min.x, mesh.bounds.min.y, mesh.bounds.min.z, 0.0f} : Core::float4{};
        record.boundsMax = bounded ? Core::float4{mesh.bounds.max.x, mesh.bounds.max.y, mesh.bounds.max.z, 0.0f} : Core::float4{1.0f, 1.0f, 1.0f, 0.0f};
        record.positions = gpu.positions.getPtr().as<float>();
        record.indices = gpu.indices.getPtr();
        record.normals = getPointer(gpu.normals, !mesh.normals.empty());
        record.tangents = getPointer(gpu.tangents, !mesh.tangents.empty());
        record.uv0 = getPointer(gpu.uvs[0], !mesh.uvs[0].empty());
        record.uv1 = getPointer(gpu.uvs[1], !mesh.uvs[1].empty());
        record.uv2 = getPointer(gpu.uvs[2], !mesh.uvs[2].empty());
        record.uv3 = getPointer(gpu.uvs[3], !mesh.uvs[3].empty());
        record.colors = getPointer(gpu.colors, !mesh.colors.empty());
        record.triangleMaterials = gpu.triangleMaterials.getPtr();
        record.vertexCount = mesh.getVertexCount();
        record.triangleCount = mesh.getTriangleCount();
        record.uvSetCount = mesh.getUvSetCount();
        record.unused = 0;
        byteSize += getBytes(gpu.positions) + getBytes(gpu.indices) + getBytes(gpu.normals) + getBytes(gpu.tangents) + getBytes(gpu.colors)
                    + getBytes(gpu.triangleMaterials);
        for (const NoorRHI::Buffer<std::uint32_t>& uv : gpu.uvs)
            byteSize += getBytes(uv);
    }
    return uploadValues<Shaders::MeshRecord>(device, records, values);
}

}
