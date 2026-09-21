#include "Raytracer.h"

#include "Materials/MaterialX/SlangMaterialCompiler.h"

#include "RealtimeRaytracer.h"

#include <map>
#include <unordered_map>

#include "Scene/GaussianInstance.h"

#include <noorrhi/interop.hpp>

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <array>
#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <glm/geometric.hpp>

#include "Logging/Log.h"
#include "Mesh/Assets/Mesh.h"
#include "Materials/Material.h"
#include "Scene/Scene.h"
#include "Scene/LightInstance.h"
#include "Scene/MeshInstance.h"
#include "Materials/Shading/ShadingTables.h"
#include "Environment/Environment.h"
#include "Camera/CameraInstance.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/RealisticCamera.h"
#include "Camera/ThinLensCamera.h"

namespace
{
constexpr float LightPi = 3.14159265358979323846f;

float lightSelectionLuminance(const glm::vec3 color)
{
    return fmaxf(glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f)), 0.0f);
}

float pointLightSelectionWeight(const PointLight& light)
{
    return lightSelectionLuminance(light.color) * fmaxf(light.intensity, 0.0f)
        * (4.0f * LightPi);
}

float spotLightSelectionWeight(const SpotLight& light)
{
    const float cone = 2.0f * LightPi * (1.0f - cosf(
        fmaxf(light.innerConeAngle, light.outerConeAngle)
            * LightPi / 180.0f));
    return lightSelectionLuminance(light.color) * fmaxf(light.intensity, 0.0f)
        * cone;
}

float rectLightSelectionWeight(const RectLight& light)
{
    const float sidedness = light.twoSided != 0 ? 2.0f : 1.0f;
    return lightSelectionLuminance(light.color) * fmaxf(light.intensity, 0.0f)
        * fmaxf(light.width * light.height, 0.0f) * LightPi * sidedness;
}

float directionalLightSelectionWeight(const DirectionalLight& light)
{
    return lightSelectionLuminance(light.color) * fmaxf(light.intensity, 0.0f);
}

// Keep texture uploads bounded by the descriptor-heap budget.  The first
// entry is reserved for the white fallback, leaving room for render targets,
// scene buffers, and repeated immutable material updates.

noorrhi::Buffer<std::byte> upload_bytes(noorrhi::Device& device, const void* data,
    const std::size_t size)
{
    if (size == 0)
        return {};
    auto buffer = device.buffer<std::byte>(size);
    buffer.upload(std::span<const std::byte>(
        static_cast<const std::byte*>(data), size));
    return buffer;
}

template<class T>
noorrhi::Buffer<std::byte> upload_value(noorrhi::Device& device, const T& value)
{
    return upload_bytes(device, &value, sizeof(value));
}

}

Raytracer::Raytracer(noorrhi::Device& device,
    const uint32_t width, const uint32_t height, const bool exportColorMemory)
    : renderWidth(std::max(width, 1u))
    , renderHeight(std::max(height, 1u))
    , imageWidth_(renderWidth)
    , imageHeight_(renderHeight)
    , gpuDevice(&device)
    , exportColorMemory(exportColorMemory)
{
    dispatchTimestamp = gpuDevice->timestamp();
    NR_LOG_INFO("graphics API raytracer: using NoorRHI API (rayQuery="
        << gpuDevice->features().ray_query << ", rayTracing="
        << gpuDevice->features().ray_tracing << ")");
    NR_LOG_INFO("graphics API raytracer: creating output images");
    createImages();
    NR_LOG_INFO("graphics API raytracer: creating upload buffers");
    lens = noorrhi::Shared<nr::graphics::Lens>(*gpuDevice);

    // An empty pointer table until the first uploadMaterials().
    const std::uint64_t noMaterial = 0;
    materials = gpuDevice->buffer<std::uint64_t>(1);
    materials.upload(std::span<const std::uint64_t>(&noMaterial, 1));
    pointLights = gpuDevice->buffer<nr::graphics::PointLight>(1);
    spotLights = gpuDevice->buffer<nr::graphics::SpotLight>(1);
    rectLights = gpuDevice->buffer<nr::graphics::RectLight>(1);
    directionalLights = gpuDevice->buffer<nr::graphics::DirectionalLight>(1);
    meshLights = gpuDevice->buffer<nr::graphics::MeshLight>(1);
    // A missing or unresolved MaterialX image must be deterministic and
    // harmless.  Keep one immutable white texel in the descriptor table and
    // map invalid scene texture references to it during material upload.
    const float whitePixel[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    whiteTexture = gpuDevice->image<std::byte>(1, 1,
        noorrhi::ImageUsage::Sampled, noorrhi::ImageFormat::Rgba32Float);
    whiteTexture.upload(std::span<const std::byte>(std::as_bytes(std::span(whitePixel))));
    // Immutable shading data shared by every closure: energy-compensation LUTs
    // and the CIE/D65 spectral tables.
    {
        const std::vector<std::uint16_t> lut = nr::shading::packEnergyLutTables();
        energyLutBuffer = upload_bytes(*gpuDevice, lut.data(),
            lut.size() * sizeof(std::uint16_t));
        data.energyLuts = energyLutBuffer.ptr().address;
        const std::vector<float> spectral = nr::shading::packSpectralTables();
        spectralTablesBuffer = upload_bytes(*gpuDevice, spectral.data(),
            spectral.size() * sizeof(float));
        data.spectralTables = spectralTablesBuffer.ptr().address;
    }
    NR_LOG_INFO("graphics API raytracer: updating descriptors");
    updateRoot();
    commit();
    NR_LOG_INFO("graphics API raytracer: ready");
}

Raytracer::~Raytracer() = default;

std::unique_ptr<Raytracer> Raytracer::create(noorrhi::Device& device,
    const uint32_t width, const uint32_t height,
    const bool exportColorMemory)
{
    // Realtime RGB is the active renderer for now. Keep the factory so the
    // spectral implementation can be reintroduced without changing hosts.
    return std::make_unique<RealtimeRaytracer>(device, width, height,
        exportColorMemory);
}

void Raytracer::createImages()
{
    const auto storage = noorrhi::ImageUsage::Storage | noorrhi::ImageUsage::Sampled;
    // Beauty is the authoritative scene-linear HDR image.  Presentation to an
    // 8-bit swapchain is a graphics API blit; offline integrations retain all
    // radiance and alpha values through readBeauty().
    const auto color_usage = exportColorMemory
        ? storage | noorrhi::ImageUsage::ExternalMemory : storage;
    colorImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, color_usage,
        noorrhi::ImageFormat::Rgba32Float);
    albedoImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
        noorrhi::ImageFormat::Rgba32Float);
    normalImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
        noorrhi::ImageFormat::Rgba32Float);
    positionImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
        noorrhi::ImageFormat::Rgba32Float);
    cryptomatteImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
        noorrhi::ImageFormat::R32Uint);
    // Per-pixel buffers are indexed with the logical width as the stride, so
    // any logical size up to the image allocation fits inside them.
    const std::size_t pixelCount = static_cast<std::size_t>(imageWidth_) * imageHeight_;
    gaussianOverdrawBuffer = gpuDevice->buffer<std::uint32_t>(pixelCount);
    accumulationBuffer = gpuDevice->buffer<noorrhi::float4>(pixelCount);
    std::vector<noorrhi::float4> clear(pixelCount, noorrhi::float4{});
    accumulationBuffer.upload(std::span<const noorrhi::float4>(clear));
}

void Raytracer::updateRoot()
{
    // Nothing is published to a heap here any more. Buffers contribute their
    // device address and images their descriptor-heap index; both are plain
    // values copied into the frame record.
    const auto address = [](const auto& buffer) -> std::uint64_t {
        return buffer ? buffer.ptr().address : 0;
    };
    const noorrhi::AccelerationStructureHandle topLevel = tlas.handle();

    data.colorImage = colorImage.storage_handle().value;
    data.albedoImage = albedoImage.storage_handle().value;
    data.normalImage = normalImage.storage_handle().value;
    data.positionImage = positionImage.storage_handle().value;
    data.cryptomatteImage = cryptomatteImage.storage_handle().value;
    data.gaussianOverdraw = address(gaussianOverdrawBuffer);
    data.lens = lens.ptr().address;
    data.scene = sceneBuffers();
    data.gaussianRecords = address(gaussianRecords_);
    data.gaussianOpacities = address(gaussianOpacities_);
    data.gaussianShCoefficients = address(gaussianShCoefficients_);
    data.gaussianInstanceOffsets = address(gaussianInstanceOffsets_);
    data.gaussianCount = gaussianCount_;
    // Only the scene-derived coefficient count lives in the low bits; the
    // shading-mode flag above it and the cutoff distance come from
    // RenderSettings and must survive a resize.
    data.gaussianShCoefficientCount =
        (data.gaussianShCoefficientCount & ~0xFFu)
        | (gaussianShCoefficientCount_ & 0xFFu);
    data.materials = address(materials);
    data.accumulation = address(accumulationBuffer);
    data.width = renderWidth;
    data.height = renderHeight;
    // The TLAS handle is its device address, converted back to a traceable
    // structure in the shader.
    data.topLevelAS = topLevel.value;
    data.pointLights = address(pointLights);
    data.spotLights = address(spotLights);
    data.rectLights = address(rectLights);
    data.directionalLights = address(directionalLights);
    data.meshLights = address(meshLights);
}

void Raytracer::resize(const uint32_t width, const uint32_t height)
{
    if (width == 0 || height == 0
        || (width == renderWidth && height == renderHeight))
        return;
    renderWidth = width;
    renderHeight = height;
    const bool reserved = reservedWidth_ != 0 && reservedHeight_ != 0;
    const bool fits = reserved
        ? width <= imageWidth_ && height <= imageHeight_
        : width == imageWidth_ && height == imageHeight_;
    if (fits)
    {
        // Only the traced rectangle changes. Work already recorded against
        // these images stays valid, so there is nothing to wait for.
        updateRoot();
        return;
    }
    reallocate(std::max(width, reservedWidth_), std::max(height, reservedHeight_));
}

void Raytracer::reserve(const uint32_t width, const uint32_t height)
{
    reservedWidth_ = width;
    reservedHeight_ = height;
    const uint32_t targetWidth = std::max(width, renderWidth);
    const uint32_t targetHeight = std::max(height, renderHeight);
    if (targetWidth != imageWidth_ || targetHeight != imageHeight_)
        reallocate(targetWidth, targetHeight);
}

void Raytracer::reallocate(const uint32_t width, const uint32_t height)
{
    // In-flight command buffers may still reference the images being replaced.
    gpuDevice->synchronize();
    imageWidth_ = width;
    imageHeight_ = height;
    createImages();
    onImageAllocationChanged();
    updateRoot();
}

void Raytracer::commit()
{
    lens.commit();
    // The Environment commits its own record; data.environment is published
    // by uploadEnvironment().
    data.lens = lens.ptr().address;
}

void Raytracer::uploadScene(Scene& scene)
{
    gpuDevice->synchronize();
    // Rebuild the material pointer table here, not just when a compilation
    // finishes: adding an object adds a material, and the hit shaders index
    // this table by the surface's material index. A table still sized to the
    // previous material count would be read out of bounds and its garbage
    // dereferenced as a device pointer. uploadMaterials() uploads lights too.
    uploadMaterials(scene);
    buildScene(scene);
    const auto address = [](const auto& buffer) -> std::uint64_t {
        return buffer ? buffer.ptr().address : 0;
    };
    data.scene = sceneBuffers();
    data.gaussianRecords = address(gaussianRecords_);
    data.gaussianOpacities = address(gaussianOpacities_);
    data.gaussianShCoefficients = address(gaussianShCoefficients_);
    data.gaussianInstanceOffsets = address(gaussianInstanceOffsets_);
    data.gaussianCount = gaussianCount_;
    data.gaussianProxyTriangleCount = gaussianProxyTriangleCount_;
    data.gaussianInstanceBase = meshInstanceCount_;
    data.gaussianShCoefficientCount = gaussianShCoefficientCount_;
    applyRenderSettings(scene.getRenderSettings());
    data.topLevelAS = tlas.handle().value;
}

bool Raytracer::updateScene(const Scene& scene, const bool updateGaussians)
{
    if (!tlas)
        return false;
    if (!updateMutableData(scene, updateGaussians))
        return false;
    data.gaussianShCoefficientCount = gaussianShCoefficientCount_;
    applyRenderSettings(scene.getRenderSettings());
    return true;
}

void Raytracer::applyRenderSettings(const RenderSettings& settings)
{
    data.gaussianShCoefficientCount =
        (data.gaussianShCoefficientCount & ~0x80000000u)
        | (settings.gaussianShadingMode == GaussianShadingMode::DirectColor
            ? 0x80000000u : 0u);
    const float cutoff = settings.gaussianCutoffSigma;
    data.gaussianCutoffDistanceSq = cutoff * cutoff;
    data.maxBounces = static_cast<std::uint32_t>(std::max(
        settings.maxBounces, 1));
    data.indirectLightClamp = settings.indirectLightClamp;
    data.transparentBackground = settings.transparentBackground ? 1u : 0u;
    data.gaussianOverdrawEnabled = rendersProxyOverdraw(settings) ? 1u : 0u;
    data.gaussianOverdrawMax = static_cast<std::uint32_t>(std::max(
        settings.gaussianProxyOverdrawMax, 1));
    data.aovEnabled = settings.aovEnabled ? 1u : 0u;
    onRenderSettingsApplied(settings);
}

void Raytracer::updateCamera(const Scene& scene)
{
    nr::graphics::Camera snapshot{};
    if (const CameraInstance* instance = scene.getRenderCamera())
    {
        const Camera* camera = instance->getCamera();
        snapshot = camera->getData();
    }
    if (const CameraInstance* instance = scene.getRenderCamera())
        if (const auto* realistic = instance->getCamera()->CastOrNullptr<RealisticCamera>())
            lens.setData(realistic->optics);
    data.camera = snapshot;
}

void Raytracer::updateLights(const Scene& scene)
{
    uploadLights(scene);
}

void Raytracer::uploadLights(const Scene& scene)
{
    // This routine is called while the renderer is idle by uploadScene. Keep
    // each replacement immutable so a dispatch cannot observe a partially
    // written typed light buffer.
    std::vector<nr::graphics::PointLight> pointRecords;
    std::vector<nr::graphics::SpotLight> spotRecords;
    std::vector<nr::graphics::RectLight> rectRecords;
    std::vector<nr::graphics::DirectionalLight> directionalRecords;
    std::vector<nr::graphics::MeshLight> meshRecords;
    pointRecords.reserve(scene.getPointLightCount());
    spotRecords.reserve(scene.getSpotLightCount());
    rectRecords.reserve(scene.getRectLightCount());
    directionalRecords.reserve(scene.getDirectionalLightCount());

    auto copyVec3 = [](float3& dst, const glm::vec3& src) {
        dst = src;
    };

    for (uint32_t i = 0; i < scene.getPointLightCount(); ++i)
    {
        const PointLight& source = scene.getPointLights()[i];
        PointLight record = source;
        copyVec3(record.position, source.position);
        copyVec3(record.color, source.color);
        record.intensity = source.intensity;
        record.selectionWeight = pointLightSelectionWeight(source);
        pointRecords.push_back(record);
    }
    for (uint32_t i = 0; i < scene.getSpotLightCount(); ++i)
    {
        const SpotLight& source = scene.getSpotLights()[i];
        SpotLight record = source;
        copyVec3(record.position, source.position);
        copyVec3(record.direction, source.direction);
        copyVec3(record.color, source.color);
        record.intensity = source.intensity;
        record.innerConeAngle = source.innerConeAngle;
        record.outerConeAngle = source.outerConeAngle;
        record.selectionWeight = spotLightSelectionWeight(source);
        spotRecords.push_back(record);
    }
    for (uint32_t i = 0; i < scene.getRectLightCount(); ++i)
    {
        const RectLight& source = scene.getRectLights()[i];
        RectLight record = source;
        copyVec3(record.position, source.position);
        copyVec3(record.direction, source.direction);
        copyVec3(record.color, source.color);
        copyVec3(record.tangent, source.tangent);
        record.intensity = source.intensity;
        record.width = source.width;
        record.height = source.height;
        record.twoSided = source.twoSided != 0 ? 1u : 0u;
        record.barnDoorAngle = source.barnDoorAngle;
        record.barnDoorLength = source.barnDoorLength;
        record.selectionWeight = rectLightSelectionWeight(source);
        rectRecords.push_back(record);
    }
    for (uint32_t i = 0; i < scene.getDirectionalLightCount(); ++i)
    {
        const DirectionalLight& source = scene.getDirectionalLights()[i];
        DirectionalLight record = source;
        copyVec3(record.direction, source.direction);
        copyVec3(record.color, source.color);
        record.intensity = source.intensity;
        record.softAngle = source.softAngle;
        record.selectionWeight = directionalLightSelectionWeight(source);
        directionalRecords.push_back(record);
    }

    // Emissive SVM programs become triangle light candidates for the spectral
    // integrator. The realtime RGB path deliberately reaches emissive meshes
    // through BSDF sampling, avoiding a CPU triangle scan and a second light
    // sampling path in the interactive shader.
    const auto meshInstances = scene.getMeshInstances();
    const auto& materials = scene.getMaterials();
    for (uint32_t instanceIndex = 0;
        supportsMeshLights()
            && instanceIndex < meshInstances.size(); ++instanceIndex)
    {
        const MeshInstance& instance = *meshInstances[instanceIndex];
        const Mesh& mesh = instance.getMesh();
        const auto& vertices = mesh.getVertices();
        const auto& indices = mesh.getIndices();
        const auto& faces = mesh.getFaces();
        const glm::mat4 transform = instance.getWorldTransform().getMatrix();
        for (uint32_t primitive = 0; primitive < faces.size()
             && primitive * 3u + 2u < indices.size(); ++primitive)
        {
            const Face face = faces[primitive];
            if (face.materialIndex < 0
                || static_cast<size_t>(face.materialIndex) >= mesh.getMaterialCount())
                continue;
            const uint32_t materialIndex = mesh.getMaterialIds()[face.materialIndex];
            if (materialIndex >= materials.size() || materials[materialIndex].mayEmit == 0u)
                continue;
            const uint32_t ia = indices[primitive * 3u];
            const uint32_t ib = indices[primitive * 3u + 1u];
            const uint32_t ic = indices[primitive * 3u + 2u];
            if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size())
                continue;
            const glm::vec3 a = glm::vec3(transform * glm::vec4(vertices[ia].position, 1.0f));
            const glm::vec3 b = glm::vec3(transform * glm::vec4(vertices[ib].position, 1.0f));
            const glm::vec3 c = glm::vec3(transform * glm::vec4(vertices[ic].position, 1.0f));
            const float area = 0.5f * glm::length(glm::cross(b - a, c - a));
            if (!(area > 0.0f) || !std::isfinite(area))
                continue;
            nr::graphics::MeshLight record{};
            record.instanceIndex = instanceIndex;
            record.primitiveIndex = primitive;
            copyVec3(record.a, a);
            copyVec3(record.b, b);
            copyVec3(record.c, c);
            record.area = area;
            record.selectionWeight = area;
            meshRecords.push_back(record);
        }
    }

    float finiteWeight = 0.0f;
    // Callers synchronize the device before scene mutations are applied, so
    // no dispatch still reads these buffers and they can be patched in place.
    const bool inPlace = lightBuffersValid_
        && pointRecords.size() == pointLightData_.size()
        && spotRecords.size() == spotLightData_.size()
        && rectRecords.size() == rectLightData_.size()
        && directionalRecords.size() == directionalLightData_.size()
        && meshRecords.size() == meshLightData_.size();
    auto upload = [this, inPlace](auto& buffer, auto& mirror, auto& records) {
        using Record = typename std::decay_t<decltype(records)>::value_type;
        if (!inPlace) {
            buffer = gpuDevice->buffer<Record>(std::max<std::size_t>(records.size(), 1u));
            if (!records.empty())
                buffer.upload(std::span<const Record>(records));
        } else {
            std::size_t first = records.size();
            std::size_t last = 0;
            for (std::size_t index = 0; index < records.size(); ++index) {
                if (std::memcmp(&records[index], &mirror[index], sizeof(Record)) == 0)
                    continue;
                first = std::min(first, index);
                last = index;
            }
            if (first < records.size())
                buffer.upload(std::span<const Record>(records.data() + first,
                    last - first + 1), first);
        }
        mirror = std::move(records);
    };
    auto accumulateWeight = [&finiteWeight](const auto& records) {
        for (const auto& record : records)
            finiteWeight += std::max(record.selectionWeight, 0.0f);
    };
    accumulateWeight(pointRecords);
    accumulateWeight(spotRecords);
    accumulateWeight(rectRecords);
    accumulateWeight(directionalRecords);
    accumulateWeight(meshRecords);

    upload(pointLights, pointLightData_, pointRecords);
    upload(spotLights, spotLightData_, spotRecords);
    upload(rectLights, rectLightData_, rectRecords);
    upload(directionalLights, directionalLightData_, directionalRecords);
    upload(meshLights, meshLightData_, meshRecords);
    lightBuffersValid_ = true;
    data.pointLights = pointLights.ptr().address;
    data.pointLightCount = static_cast<std::uint32_t>(pointLightData_.size());
    data.spotLights = spotLights.ptr().address;
    data.spotLightCount = static_cast<std::uint32_t>(spotLightData_.size());
    data.rectLights = rectLights.ptr().address;
    data.rectLightCount = static_cast<std::uint32_t>(rectLightData_.size());
    data.directionalLights = directionalLights.ptr().address;
    data.directionalLightCount = static_cast<std::uint32_t>(directionalLightData_.size());
    data.meshLights = meshLights.ptr().address;
    data.meshLightCount = static_cast<std::uint32_t>(meshLightData_.size());
    data.lightFiniteWeight = finiteWeight;
    onLightsUploaded();
}

void Raytracer::uploadTextures(Scene& scene)
{
    for (Texture& texture : scene.getTextures())
    {
        if (texture)
            continue;
        try {
            texture.upload(*gpuDevice);
        } catch (const std::exception& error) {
            NR_LOG_WARN("Texture upload failed; sampling white instead for "
                << texture.getName() << ": " << error.what());
        }
    }
}

void Raytracer::uploadEnvironment(Scene& scene)
{
    // The Environment owns its HDRI and CDF images and uploads them when the
    // texture changes. Per-frame scalar edits only republish its small record.
    ::Environment& environment = scene.getEnvironment();
    uploadTextures(scene);
    if (!environment.hdriImage && environment.getTextureIndex() >= 0
        && static_cast<std::size_t>(environment.getTextureIndex()) < scene.getTextures().size())
        environment.uploadImages(*gpuDevice,
            &scene.getTextures()[environment.getTextureIndex()]);
    else
        environment.uploadRecord(*gpuDevice);
    data.environment = environment.ptr().address;
}

void Raytracer::uploadMaterials(Scene& scene)
{
    uploadTextures(scene);
    const auto resolveTexture = [this, &scene](const std::uint32_t index) -> std::uint32_t {
        if (index < scene.getTextures().size()
            && scene.getTextures()[index])
            return scene.getTextures()[index].sampledHandle().value;
        return whiteTexture.sampled_handle().value;
    };
    const std::size_t shaderCount = materialShaders_.size();
    for (Material& material : scene.getMaterials()) {
        if (material)
            continue;
        std::uint32_t shaderIndex = ~0u;
        if (const auto& shader = material.shaderProgram.shader) {
            const auto found = std::ranges::find(materialShaderPrograms_, shader);
            shaderIndex = static_cast<std::uint32_t>(found - materialShaderPrograms_.begin());
            if (found == materialShaderPrograms_.end()) {
                materialShaderPrograms_.push_back(shader);
                materialShaders_.push_back(gpuDevice->create_shader(
                    std::as_bytes(std::span(shader->spirv)), "main"));
            }
        }
        material.upload(*gpuDevice, resolveTexture, shaderIndex);
    }
    if (materialShaders_.size() != shaderCount)
        onMaterialShadersChanged(materialShaders_);

    std::vector<std::uint64_t> pointers;
    pointers.reserve(std::max<std::size_t>(scene.getMaterials().size(), 1u));
    for (const Material& material : scene.getMaterials())
        pointers.push_back(material ? material.ptr().address : 0);
    if (pointers.empty())
        pointers.push_back(0);
    materials = gpuDevice->buffer<std::uint64_t>(pointers.size());
    materials.upload(std::span<const std::uint64_t>(pointers));

    data.materials = materials.ptr().address;
    data.accumulation = accumulationBuffer.ptr().address;
    uploadLights(scene);
}

void Raytracer::render(const uint32_t frameIndex, const uint32_t sampleIndex)
{
    data.frameIndex = frameIndex;
    data.sampleIndex = sampleIndex;
    data.width = renderWidth;
    data.height = renderHeight;

    // Inside a noorrhi::Frame this batches into the frame's command buffer; with
    // no frame open it is submitted on its own, which is the offline path.
    gpuDevice->measure(dispatchTimestamp, [this] { renderImpl(); });
}

void Raytracer::renderImpl()
{
    pipeline.trace({renderWidth, renderHeight, 1}, data);
}

double Raytracer::lastDispatchMilliseconds()
{
    return dispatchTimestamp.milliseconds();
}

std::vector<std::byte> Raytracer::readColor() {
    const auto beauty = readBeauty();
    std::vector<std::byte> result(beauty.size() * 4u);
    for (std::size_t i = 0; i < beauty.size(); ++i) {
        const auto encode = [](const float value) -> std::byte {
            return static_cast<std::byte>(static_cast<unsigned char>(
                std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f));
        };
        result[4 * i] = encode(beauty[i].z);
        result[4 * i + 1] = encode(beauty[i].y);
        result[4 * i + 2] = encode(beauty[i].x);
        result[4 * i + 3] = encode(beauty[i].w);
    }
    return result;
}

template<class T>
std::vector<T> Raytracer::cropToRender(std::vector<T> pixels) const
{
    // Downloads cover the whole allocation, where texel (x, y) is at
    // y * imageWidth + x. Keep only the logical rectangle, row by row.
    if (imageWidth_ == renderWidth && imageHeight_ == renderHeight)
        return pixels;
    std::vector<T> result(static_cast<std::size_t>(renderWidth) * renderHeight);
    for (uint32_t y = 0; y < renderHeight; ++y)
        std::copy_n(pixels.begin() + static_cast<std::ptrdiff_t>(y) * imageWidth_,
            renderWidth, result.begin() + static_cast<std::ptrdiff_t>(y) * renderWidth);
    return result;
}

std::vector<noorrhi::float4> Raytracer::readBeauty()
{
    std::vector<noorrhi::float4> result(static_cast<std::size_t>(imageWidth_)
        * imageHeight_);
    colorImage.download(std::as_writable_bytes(std::span(result)));
    return cropToRender(std::move(result));
}

std::vector<std::uint32_t> Raytracer::readCryptomatte()
{
    std::vector<std::uint32_t> result(static_cast<std::size_t>(imageWidth_)
        * imageHeight_);
    std::vector<std::byte> bytes(result.size() * sizeof(result.front()));
    cryptomatteImage.download(std::span<std::byte>(bytes));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return cropToRender(std::move(result));
}

std::vector<noorrhi::float4> Raytracer::readPosition()
{
    std::vector<noorrhi::float4> result(static_cast<std::size_t>(imageWidth_)
        * imageHeight_);
    std::vector<std::byte> bytes(result.size() * sizeof(result.front()));
    positionImage.download(std::span<std::byte>(bytes));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return cropToRender(std::move(result));
}

std::uint32_t Raytracer::readCryptomatteAt(const uint32_t x, const uint32_t y)
{
    std::uint32_t id = ~0u;
    if (x >= renderWidth || y >= renderHeight)
        return id;
    std::array<std::byte, sizeof(id)> bytes{};
    cryptomatteImage.download_region(x, y, 1, 1, std::span<std::byte>(bytes));
    std::memcpy(&id, bytes.data(), bytes.size());
    return id;
}

noorrhi::float4 Raytracer::readPositionAt(const uint32_t x, const uint32_t y)
{
    noorrhi::float4 position{};
    if (x >= renderWidth || y >= renderHeight)
        return position;
    std::array<std::byte, sizeof(position)> bytes{};
    positionImage.download_region(x, y, 1, 1, std::span<std::byte>(bytes));
    std::memcpy(&position, bytes.data(), bytes.size());
    return position;
}

namespace
{

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


void Raytracer::buildScene(Scene& scene)
{
    sourceMeshes_.clear();
    gaussianAssets_.clear();
    gaussianAssetCounts_.clear();
    gaussianTransforms.clear();
    tlasInstances.clear();
    for (Mesh& mesh : scene.getMeshes())
        mesh.upload(*gpuDevice);
    std::unordered_map<const Mesh*, uint32_t> meshIndices;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMesh())
            continue;
        const Mesh* asset = &instance->getMesh();
        if (!meshIndices.contains(asset))
        {
            const uint32_t index = static_cast<uint32_t>(sourceMeshes_.size());
            meshIndices.emplace(asset, index);
            buildMesh(*asset);
        }
    }
    buildGaussians(scene);
    buildTopLevel(scene);
    buildSceneData(scene);
}

void Raytracer::buildGaussians(const Scene& scene)
{
    std::vector<Gaussian> records;
    std::vector<float> opacities;
    std::vector<half> shCoefficients;
    std::vector<uint32_t> instanceOffsets;
    const uint32_t coefficientCount = sphericalHarmonicsCoefficientCount(
        scene.getRenderSettings().gaussianRenderSphericalHarmonics);
    gaussianShCoefficientCount_ = coefficientCount;

    for (const auto& instance : scene.getGaussianInstances())
    {
        if (!instance || !instance->hasGaussianAsset())
            continue;
        instanceOffsets.push_back(static_cast<uint32_t>(records.size()));
        const auto& source = instance->getGaussianAsset().getGaussians();
        gaussianAssets_.push_back(&instance->getGaussianAsset());
        gaussianAssetCounts_.push_back(source.size());
        const glm::mat4 world = instance->getWorldTransform().getMatrix();
        const glm::mat3 linear(world);
        const size_t recordBegin = records.size();
        records.insert(records.end(), source.begin(), source.end());
        // GaussianAsset stores R*S and its center in object space.  The
        // acceleration path historically applied the instance transform via
        // the proxy TLAS transform; graphics API's compact Gaussian arrays are
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
            shCoefficients.resize(oldSize + coefficientCount * 3u, half{0});
            for (uint32_t coefficient = 0; coefficient < available; ++coefficient)
            {
                std::memcpy(shCoefficients.data() + oldSize + coefficient * 3u,
                    gaussian.sphericalHarmonics.values.data() + coefficient * 3u,
                    sizeof(half) * 3u);
            }
        }

        // Use one shared tiny proxy BLAS for every Gaussian instance
        // for every Gaussian. Expanding every transformed proxy into unique
        // vertices turns a 1.6M-splat level-2 scene into 533M triangles.
        gaussianTransforms.reserve(gaussianTransforms.size() + source.size());
        for (std::size_t local = 0; local < source.size(); ++local)
        {
            const Gaussian& gaussian = records[recordBegin + local];
            noorrhi::float4x4 transform{};
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
    std::vector<noorrhi::float3> positions;
    positions.reserve(unitProxy.vertices.size());
    for (const glm::vec3 vertex : unitProxy.vertices)
        positions.push_back({vertex.x * proxyScale, vertex.y * proxyScale,
            vertex.z * proxyScale});
    gaussianProxy.positions = (*gpuDevice).buffer<noorrhi::float3>(positions.size());
    gaussianProxy.indices = (*gpuDevice).buffer<std::uint32_t>(unitProxy.indices.size());
    gaussianProxy.positions.upload(std::span<const noorrhi::float3>(positions));
    gaussianProxy.indices.upload(std::span<const std::uint32_t>(unitProxy.indices));
    const noorrhi::TriangleGeometry proxyGeometry{gaussianProxy.positions.ptr(),
        gaussianProxy.indices.ptr(), gaussianProxyTriangleCount_, false};
    gaussianProxy.blas = (*gpuDevice).build_blas(
        std::span<const noorrhi::TriangleGeometry>(&proxyGeometry, 1));

    const auto publish = [this]<class T>(noorrhi::Buffer<T>& buffer, const std::vector<T>& values) {
        if (values.empty()) { buffer = {}; return; }
        buffer = gpuDevice->buffer<T>(values.size());
        buffer.upload(std::span<const T>(values));
    };
    gaussianRecordData_ = std::move(records);
    gaussianOpacityData_ = std::move(opacities);
    gaussianShCoefficientData_ = std::move(shCoefficients);
    publish(gaussianRecords_, gaussianRecordData_);
    publish(gaussianOpacities_, gaussianOpacityData_);
    publish(gaussianShCoefficients_, gaussianShCoefficientData_);
    publish(gaussianInstanceOffsets_, instanceOffsets);
}

void Raytracer::buildSceneData(const Scene& scene)
{
    std::vector<const Mesh*> assets;
    std::unordered_map<const Mesh*, uint32_t> meshIndices;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMesh())
            continue;
        const Mesh* asset = &instance->getMesh();
        if (!meshIndices.contains(asset))
        {
            meshIndices.emplace(asset, static_cast<uint32_t>(assets.size()));
            assets.push_back(asset);
        }
    }

    std::vector<std::uint64_t> meshRecords;
    meshRecords.reserve(assets.size());
    for (const Mesh* asset : assets)
    {
        if (!asset->vertexBuffer || !asset->indexBuffer || !asset->blas)
            throw std::runtime_error("Mesh GPU data was not initialized: " + asset->getName());
        meshRecords.push_back(asset->ptr().address);
    }
    std::vector<nr::graphics::Instance> instanceRecords;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMesh())
            continue;
        nr::graphics::Instance record{};
        const glm::mat4 transform = instance->getWorldTransform().getMatrix();
        // Slang is compiled row-major. Store the conventional row-major
        // affine matrix while GLM remains column-major on the host.
        for (uint32_t row = 0; row < 3; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                record.objectToWorld[row * 4u + column] = transform[column][row];
        record.meshIndex = meshIndices.at(&instance->getMesh());
        instanceRecords.push_back(record);
    }

    instanceData_ = std::move(instanceRecords);
    meshRecordData_ = std::move(meshRecords);
    uploadMeshRecords();
    uploadInstances();
}

void Raytracer::uploadMeshRecords()
{
    if (meshRecordData_.empty()) { meshRecords_ = {}; return; }
    if (!meshRecords_ || meshRecords_.size() != meshRecordData_.size())
        meshRecords_ = gpuDevice->buffer<std::uint64_t>(meshRecordData_.size());
    meshRecords_.upload(std::span<const std::uint64_t>(meshRecordData_));
}

void Raytracer::uploadInstances()
{
    if (instanceData_.empty()) { instances_ = {}; return; }
    if (!instances_ || instances_.size() != instanceData_.size())
        instances_ = gpuDevice->buffer<nr::graphics::Instance>(instanceData_.size());
    instances_.upload(std::span<const nr::graphics::Instance>(instanceData_));
}

nr::graphics::Scene Raytracer::sceneBuffers() const
{
    const auto address = []<class T>(const noorrhi::Buffer<T>& buffer) {
        return buffer ? buffer.ptr().address : std::uint64_t{0};
    };
    return {
        address(meshRecords_),
        address(instances_),
        static_cast<uint32_t>(meshRecordData_.size()),
        static_cast<uint32_t>(instanceData_.size()),
    };
}

uint32_t Raytracer::buildMesh(const Mesh& asset)
{
    if (!asset.blas)
        throw std::runtime_error("Mesh GPU data was not initialized: " + asset.getName());
    sourceMeshes_.push_back(&asset);
    return static_cast<uint32_t>(sourceMeshes_.size() - 1);
}

void Raytracer::buildTopLevel(const Scene& scene)
{
    std::unordered_map<const Mesh*, uint32_t> meshIndices;
    tlasInstances.clear();
    meshInstanceAssetIndices.clear();
    uint32_t meshInstanceIndex = 0;
    for (const auto& instance : scene.getMeshInstances())
    {
        if (!instance || !instance->hasMesh())
            continue;
        const Mesh* asset = &instance->getMesh();
        auto found = meshIndices.find(asset);
        if (found == meshIndices.end())
        {
            // The build order is the same as the first pass. Reconstructing
            // this map from the scene keeps the records compact without
            // exposing mutable renderer state through Scene.
            const uint32_t index = static_cast<uint32_t>(meshIndices.size());
            found = meshIndices.emplace(asset, index).first;
        }
        tlasInstances.push_back({asset->blas,
            instance->getWorldTransform().getGpuTransform(),
            meshInstanceIndex++, 0u,
            static_cast<std::uint8_t>(nr::graphics::RaytracingMaskMesh)});
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
            0u, 2u, static_cast<std::uint8_t>(nr::graphics::RaytracingMaskGaussian)});
    }
    meshInstanceCount_ = meshInstanceIndex;
    gaussianInstanceCount_ = static_cast<uint32_t>(scene.getGaussianInstances().size());
    instanceCount_ = static_cast<uint32_t>(tlasInstances.size());
    if (tlasInstances.empty()) {
        // Publishing an empty scene must retire the previous GPU snapshot.
        // The ray-generation shader treats a null TLAS as an environment miss.
        tlas = {};
        return;
    }
    tlas = (*gpuDevice).build_tlas(std::span<const noorrhi::Instance>(tlasInstances));
}

bool Raytracer::updateMutableData(const Scene& scene, const bool updateGaussians)
{
    // Validate every topology identity before touching any live host/GPU data.
    uint32_t meshCount = 0;
    for (const auto& instance : scene.getMeshInstances()) {
        if (!instance || !instance->hasMesh()) continue;
        if (meshCount >= meshInstanceAssetIndices.size()
            || &instance->getMesh() != sourceMeshes_[meshInstanceAssetIndices[meshCount]])
            return false;
        ++meshCount;
    }
    const auto& gaussianInstances = scene.getGaussianInstances();
    if (meshCount != meshInstanceCount_ || gaussianInstances.size() != gaussianAssets_.size())
        return false;
    for (std::size_t i = 0; i < gaussianInstances.size(); ++i) {
        const auto& instance = gaussianInstances[i];
        if (!instance || !instance->hasGaussianAsset()
            || &instance->getGaussianAsset() != gaussianAssets_[i]
            || instance->getGaussianAsset().getGaussians().size() != gaussianAssetCounts_[i])
            return false;
    }
    const auto coefficientCount = sphericalHarmonicsCoefficientCount(
        scene.getRenderSettings().gaussianRenderSphericalHarmonics);
    if (gaussianCount_ && (coefficientCount != gaussianShCoefficientCount_
        || static_cast<uint32_t>(scene.getRenderSettings().gaussianProxyType) != gaussianProxyType_
        || scene.getRenderSettings().gaussianCutoffSigma != gaussianCutoffSigma_))
        return false;

    const auto assign = []<class T>(std::vector<T>& owner, std::size_t index, const T& value) {
        if (std::memcmp(&owner[index], &value, sizeof(T)) == 0) return false;
        owner[index] = value;
        return true;
    };
    bool refit = false;
    for (std::size_t index = 0; index < sourceMeshes_.size(); ++index) {
        assign(meshRecordData_, index, sourceMeshes_[index]->ptr().address);
    }
    uint32_t meshIndex = 0;
    for (const auto& instance : scene.getMeshInstances()) {
        if (!instance || !instance->hasMesh()) continue;
        nr::graphics::Instance record{};
        const glm::mat4 transform = instance->getWorldTransform().getMatrix();
        for (uint32_t row = 0; row < 3; ++row)
            for (uint32_t column = 0; column < 4; ++column)
                record.objectToWorld[row * 4u + column] = transform[column][row];
        record.meshIndex = meshInstanceAssetIndices[meshIndex];
        if (assign(instanceData_, meshIndex, record)) {
            tlasInstances[meshIndex].transform = instance->getWorldTransform().getGpuTransform();
            refit = true;
        }
        tlasInstances[meshIndex].blas = instance->getMesh().blas;
        refit = true;
        ++meshIndex;
    }

    if (updateGaussians) {
        std::size_t index = 0;
        for (const auto& instance : gaussianInstances) {
            const auto& source = instance->getGaussianAsset().getGaussians();
            const glm::mat4 world = instance->getWorldTransform().getMatrix();
            const glm::mat3 linear(world);
            for (const Gaussian& original : source) {
                Gaussian record = original;
                for (uint32_t column = 0; column < 3; ++column)
                    record.transform[column] = linear * original.transform[column];
                record.transform[3] = glm::vec3(world * glm::vec4(original.transform[3], 1.0f));
                assign(gaussianRecordData_, index, record);
                assign(gaussianOpacityData_, index, original.opacity);
                const auto available = std::min(original.sphericalHarmonics.count, coefficientCount);
                for (uint32_t coefficient = 0; coefficient < coefficientCount * 3u; ++coefficient) {
                    const half value = coefficient < available * 3u
                        ? original.sphericalHarmonics.values[coefficient] : half{0};
                    assign(gaussianShCoefficientData_, index * coefficientCount * 3u + coefficient, value);
                }
                noorrhi::float4x4 transform{};
                for (uint32_t row = 0; row < 3; ++row) {
                    for (uint32_t column = 0; column < 3; ++column)
                        transform.values[row][column] = record.transform[column][row];
                    transform.values[row][3] = record.transform[3][row];
                }
                transform.values[3][3] = 1.0f;
                auto& previous = tlasInstances[meshInstanceCount_ + index].transform;
                if (std::memcmp(&previous, &transform, sizeof(transform)) != 0) {
                    previous = transform;
                    refit = true;
                }
                ++index;
            }
        }
    }
    uploadInstances();
    uploadMeshRecords();
    if (!gaussianRecordData_.empty()) {
        gaussianRecords_.upload(std::span<const Gaussian>(gaussianRecordData_));
        gaussianOpacities_.upload(std::span<const float>(gaussianOpacityData_));
        gaussianShCoefficients_.upload(std::span<const half>(gaussianShCoefficientData_));
    }
    if (tlas && refit)
        (*gpuDevice).update_tlas(tlas, std::span<const noorrhi::Instance>(tlasInstances));
    return true;
}
