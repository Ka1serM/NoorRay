#include "RaytracerRenderer.h"

#include <gpu/interop.hpp>

#include <algorithm>
#include <cstring>
#include <span>
#include <stdexcept>

#include "Log.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Materials/Shading/Material.h"
#include "Scene/Scene.h"
#include "Scene/Objects/LightInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "EnvironmentSnapshot.h"
#include "ShadingTables.h"
#include "Scene/Resources/Environment.h"
#include "TriangleScene.h"
#include "VulkanScene.h"

namespace
{
alignas(uint32_t) constexpr unsigned char raygenSpv[] = {
    #embed "../Shaders/Raytracer/Raytracer.spv"
};
alignas(uint32_t) constexpr unsigned char missSpv[] = {
    #embed "../Shaders/Raytracer/RaytracingMiss.spv"
};
alignas(uint32_t) constexpr unsigned char hitSpv[] = {
    #embed "../Shaders/Raytracer/RaytracingHit.spv"
};
alignas(uint32_t) constexpr unsigned char emissionHitSpv[] = {
    #embed "../Shaders/Raytracer/EmissionHit.spv"
};
alignas(uint32_t) constexpr unsigned char opacityAnyHitSpv[] = {
    #embed "../Shaders/Raytracer/OpacityAnyHit.spv"
};
alignas(uint32_t) constexpr unsigned char gaussianAnyHitSpv[] = {
    #embed "../Shaders/Raytracer/GaussianAnyHit.spv"
};
alignas(uint32_t) constexpr unsigned char gaussianHitSpv[] = {
    #embed "../Shaders/Raytracer/GaussianHit.spv"
};
constexpr std::size_t raygenSpvLength = sizeof(raygenSpv);
// Keep texture uploads bounded by the descriptor-heap budget.  The first
// entry is reserved for the white fallback, leaving room for render targets,
// scene buffers, and repeated immutable material updates.
constexpr std::size_t maxSceneTextures = 1024;

gpu::Buffer<std::byte> upload_bytes(gpu::Device& device, const void* data,
    const std::size_t size)
{
    if (size == 0)
        return {};
    auto buffer = device.buffer<std::byte>(size);
    device.upload(buffer, std::span<const std::byte>(
        static_cast<const std::byte*>(data), size));
    return buffer;
}

template<class T>
gpu::Buffer<std::byte> upload_value(gpu::Device& device, const T& value)
{
    return upload_bytes(device, &value, sizeof(value));
}

}

VulkanRaytracer::VulkanRaytracer(gpu::Device& device,
    const uint32_t width, const uint32_t height, const bool buildSmokeScene,
    const bool exportColorMemory)
    : renderWidth(std::max(width, 1u))
    , renderHeight(std::max(height, 1u))
    , gpuDevice(&device)
    , exportColorMemory(exportColorMemory)
{
    dispatchTimestamp = gpuDevice->timestamp();
    LOG_INFO("Vulkan raytracer: using gpu API (rayQuery="
        << gpuDevice->features().ray_query << ", rayTracing="
        << gpuDevice->features().ray_tracing << ")");
    LOG_INFO("Vulkan raytracer: creating ray-tracing pipeline");
    createPipeline();
    LOG_INFO("Vulkan raytracer: creating output images");
    createImages();
    if (buildSmokeScene)
    {
        LOG_INFO("Vulkan raytracer: creating smoke scene");
        triangleScene = std::make_unique<TriangleScene>(*gpuDevice);
    }
    LOG_INFO("Vulkan raytracer: creating upload buffers");
    const nr::optics::LensSnapshot emptyLens{};
    const VulkanCameraSnapshot emptyCamera{};
    const Material emptyMaterial{};
    const uint32_t emptyWord = 0u;
    const VulkanLightHeader emptyLights{};
    lensBuffer = upload_value(*gpuDevice, emptyLens);
    cameraBuffer = upload_value(*gpuDevice, emptyCamera);
    materialBuffer = upload_value(*gpuDevice, emptyMaterial);
    svmWordsBuffer = upload_value(*gpuDevice, emptyWord);
    lightsBuffer = upload_value(*gpuDevice, emptyLights);
    // A missing or unresolved MaterialX image must be deterministic and
    // harmless.  Keep one immutable white texel in the descriptor table and
    // map invalid scene texture references to it during material upload.
    const float whitePixel[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    textureImages.emplace_back(gpuDevice->image<std::byte>(1, 1,
        gpu::ImageUsage::Sampled, gpu::ImageFormat::Rgba32Float));
    gpuDevice->upload(textureImages.back(),
        std::span<const std::byte>(std::as_bytes(std::span(whitePixel))));
    // Immutable shading data shared by every closure: energy-compensation LUTs
    // and the CIE/D65 spectral tables.
    {
        const std::vector<std::uint16_t> lut = nr::vulkan::packEnergyLutTables();
        energyLutBuffer = upload_bytes(*gpuDevice, lut.data(),
            lut.size() * sizeof(std::uint16_t));
        params.energyLutBuffer =
            static_cast<std::uint32_t>(energyLutBuffer.handle().value);
        const std::vector<float> spectral = nr::vulkan::packSpectralTables();
        spectralTablesBuffer = upload_bytes(*gpuDevice, spectral.data(),
            spectral.size() * sizeof(float));
        params.spectralTables =
            static_cast<std::uint32_t>(spectralTablesBuffer.handle().value);
    }
    LOG_INFO("Vulkan raytracer: updating descriptors");
    updateDescriptors();
    LOG_INFO("Vulkan raytracer: ready");
}

VulkanRaytracer::~VulkanRaytracer() = default;

void VulkanRaytracer::createPipeline()
{
    const auto raygen_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(raygenSpv), raygenSpvLength);
    const auto miss_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(missSpv), sizeof(missSpv));
    const auto hit_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(hitSpv), sizeof(hitSpv));
    const auto emission_hit_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(emissionHitSpv), sizeof(emissionHitSpv));
    const auto opacity_any_hit_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(opacityAnyHitSpv), sizeof(opacityAnyHitSpv));
    const auto gaussian_any_hit_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(gaussianAnyHitSpv), sizeof(gaussianAnyHitSpv));
    const auto gaussian_hit_bytes = std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(gaussianHitSpv), sizeof(gaussianHitSpv));
    raygenShader = gpuDevice->create_shader(raygen_bytes, "main");
    missShader = gpuDevice->create_shader(miss_bytes, "main");
    hitShader = gpuDevice->create_shader(hit_bytes, "main");
    emissionHitShader = gpuDevice->create_shader(emission_hit_bytes, "main");
    opacityAnyHitShader = gpuDevice->create_shader(opacity_any_hit_bytes, "main");
    gaussianAnyHitShader = gpuDevice->create_shader(gaussian_any_hit_bytes, "main");
    gaussianHitShader = gpuDevice->create_shader(gaussian_hit_bytes, "main");
    // Two ray types per geometry: primary/shadow and emission lookup.
    // Gaussian uses its normal stochastic any-hit for both ray types; a
    // Gaussian closest hit cannot be mistaken for mesh emission (hit == 3).
    pipeline = gpuDevice->ray_tracing({raygenShader, {missShader},
        {hitShader, emissionHitShader, gaussianHitShader, gaussianHitShader},
        {opacityAnyHitShader, opacityAnyHitShader,
            gaussianAnyHitShader, gaussianAnyHitShader}, {}});
}

void VulkanRaytracer::createImages()
{
    const auto storage = gpu::ImageUsage::Storage | gpu::ImageUsage::Sampled;
    // Beauty is the authoritative scene-linear HDR image.  Presentation to an
    // 8-bit swapchain is a Vulkan blit; offline integrations retain all
    // radiance and alpha values through readBeauty().
    const auto color_usage = exportColorMemory
        ? storage | gpu::ImageUsage::ExternalMemory : storage;
    colorImage = gpuDevice->image<std::byte>(renderWidth, renderHeight, color_usage,
        gpu::ImageFormat::Rgba32Float);
    albedoImage = gpuDevice->image<std::byte>(renderWidth, renderHeight, storage,
        gpu::ImageFormat::Rgba32Float);
    normalImage = gpuDevice->image<std::byte>(renderWidth, renderHeight, storage,
        gpu::ImageFormat::Rgba32Float);
    positionImage = gpuDevice->image<std::byte>(renderWidth, renderHeight, storage,
        gpu::ImageFormat::Rgba32Float);
    cryptomatteImage = gpuDevice->image<std::byte>(renderWidth, renderHeight, storage,
        gpu::ImageFormat::R32Uint);
    gaussianOverdrawBuffer = gpuDevice->buffer<std::uint32_t>(
        static_cast<std::size_t>(renderWidth) * renderHeight);
    accumulationBuffer = gpuDevice->buffer<gpu::float4>(
        static_cast<std::size_t>(renderWidth) * renderHeight);
    std::vector<gpu::float4> clear(static_cast<std::size_t>(renderWidth)
        * renderHeight, gpu::float4{});
    gpuDevice->upload(accumulationBuffer, std::span<const gpu::float4>(clear));
}

void VulkanRaytracer::updateDescriptors()
{
    const auto index = [](const auto handle) {
        return static_cast<std::uint32_t>(handle.value);
    };
    // gpu::Image already owns a descriptor slot for each role it was created
    // with, so the storage/sampled handles are the heap indices the shader
    // needs - there is nothing to allocate or write here.
    const gpu::AccelerationStructureHandle topLevel = nativeScene
        ? nativeScene->topLevelHandle()
        : triangleScene ? triangleScene->topLevelHandle() : gpu::AccelerationStructureHandle{};

    params.colorImage = index(colorImage.storage_handle());
    params.albedoImage = index(albedoImage.storage_handle());
    params.normalImage = index(normalImage.storage_handle());
    params.positionImage = index(positionImage.storage_handle());
    params.cryptomatteImage = index(cryptomatteImage.storage_handle());
    params.gaussianOverdrawImage = index(gaussianOverdrawBuffer.handle());
    const gpu::ResourceHandle lens = lensBuffer.handle();
    const gpu::ResourceHandle camera = cameraBuffer.handle();
    params.lensBuffer = index(lens);
    params.lensAddress = lens.value;
    params.cameraAddress = camera.value;
    params.sceneBuffer = nativeScene && nativeScene->sceneData()
        ? index(nativeScene->sceneData().handle()) : ~0u;
    params.gaussianRecords = nativeScene && nativeScene->gaussianRecords()
        ? index(nativeScene->gaussianRecords().handle()) : ~0u;
    params.gaussianOpacities = nativeScene && nativeScene->gaussianOpacities()
        ? index(nativeScene->gaussianOpacities().handle()) : ~0u;
    params.gaussianShCoefficients = nativeScene && nativeScene->gaussianShCoefficients()
        ? index(nativeScene->gaussianShCoefficients().handle()) : ~0u;
    params.gaussianInstanceOffsets = nativeScene && nativeScene->gaussianInstanceOffsets()
        ? index(nativeScene->gaussianInstanceOffsets().handle()) : ~0u;
    params.gaussianCount = nativeScene ? nativeScene->gaussianCount() : 0u;
    // Only the scene-derived coefficient count lives in the low bits; the
    // shading-mode flag above it and the cutoff distance come from
    // RenderSettings and must survive a resize.
    params.gaussianShCoefficientCount =
        (params.gaussianShCoefficientCount & ~0xFFu)
        | (nativeScene ? (nativeScene->gaussianShCoefficientCount() & 0xFFu) : 0u);
    // These two legacy-named fields carry descriptor-heap indices for the
    // native SVM records and word stream. Their width/position is preserved
    // so existing FrameParams consumers remain ABI-compatible.
    const gpu::ResourceHandle material = materialBuffer.handle();
    params.materialsAddress = material.value;
    params.svmWords = index(svmWordsBuffer ? svmWordsBuffer.handle() : material);
    params.accumulationBuffer = index(accumulationBuffer.handle());
    params.width = renderWidth;
    params.height = renderHeight;
    params.exposure = 0.0f;
    params.topLevelAS = topLevel ? static_cast<std::uint32_t>(topLevel.value) : ~0u;
    // sceneAddress is unused by the native shader path; retain its 64-bit
    // slot as the packed-light descriptor index so the established 128-byte
    // push ABI does not grow for analytic lights.
    params.sceneAddress = lightsBuffer.handle().value;
}

void VulkanRaytracer::resize(const uint32_t width, const uint32_t height)
{
    if (width == 0 || height == 0
        || (width == renderWidth && height == renderHeight))
        return;
    gpuDevice->synchronize();
    renderWidth = width;
    renderHeight = height;
    createImages();
    updateDescriptors();
}

void VulkanRaytracer::uploadLensSnapshot(const nr::optics::LensSnapshot& lens)
{
    gpuDevice->synchronize();
    const auto bytes = std::as_bytes(std::span(&lens, 1));
    gpuDevice->upload(lensBuffer, bytes);
    params.lensAddress = lensBuffer.handle().value;
    params.lensBuffer = static_cast<std::uint32_t>(params.lensAddress);
}

void VulkanRaytracer::uploadCameraSnapshot(
    const VulkanCameraSnapshot& camera)
{
    gpuDevice->synchronize();
    const auto bytes = std::as_bytes(std::span(&camera, 1));
    gpuDevice->upload(cameraBuffer, bytes);
    // The public FrameParams field is 64-bit by design. The native descriptor
    // model uses its low 32 bits as the heap index for this small immutable
    // record; the high bits remain zero and preserve the ABI width.
    params.cameraAddress = cameraBuffer.handle().value;
    params.exposure = camera.exposure;
}

void VulkanRaytracer::uploadScene(const Scene& scene)
{
    gpuDevice->synchronize();
    uploadLights(scene);
    auto replacement = std::make_unique<VulkanScene>(*gpuDevice, scene);
    nativeScene = std::move(replacement);
    params.sceneBuffer = nativeScene->sceneData()
        ? static_cast<std::uint32_t>(nativeScene->sceneData().handle().value) : ~0u;
    params.gaussianRecords = nativeScene->gaussianRecords()
        ? static_cast<std::uint32_t>(nativeScene->gaussianRecords().handle().value) : ~0u;
    params.gaussianOpacities = nativeScene->gaussianOpacities()
        ? static_cast<std::uint32_t>(nativeScene->gaussianOpacities().handle().value) : ~0u;
    params.gaussianShCoefficients = nativeScene->gaussianShCoefficients()
        ? static_cast<std::uint32_t>(nativeScene->gaussianShCoefficients().handle().value) : ~0u;
    params.gaussianInstanceOffsets = nativeScene->gaussianInstanceOffsets()
        ? static_cast<std::uint32_t>(nativeScene->gaussianInstanceOffsets().handle().value) : ~0u;
    params.gaussianCount = nativeScene->gaussianCount();
    params.gaussianProxyTriangleCount = nativeScene->gaussianProxyTriangleCount();
    params.gaussianInstanceBase = nativeScene->meshInstanceCount();
    params.gaussianShCoefficientCount = nativeScene->gaussianShCoefficientCount();
    applyRenderSettings(scene.getRenderSettings());
    const gpu::AccelerationStructureHandle topLevel = nativeScene->topLevelHandle();
    if (!topLevel)
    {
        params.topLevelAS = ~0u;
        return;
    }
    params.topLevelAS = static_cast<std::uint32_t>(topLevel.value);
}

bool VulkanRaytracer::updateScene(const Scene& scene, const bool updateGaussians)
{
    if (!nativeScene)
        return false;
    gpuDevice->synchronize();
    if (!nativeScene->updateMutableData(scene, updateGaussians))
        return false;
    params.gaussianShCoefficientCount = nativeScene->gaussianShCoefficientCount();
    applyRenderSettings(scene.getRenderSettings());
    return true;
}

void VulkanRaytracer::applyRenderSettings(const RenderSettings& settings)
{
    params.gaussianShCoefficientCount =
        (params.gaussianShCoefficientCount & ~0x80000000u)
        | (settings.gaussianShadingMode == GaussianShadingMode::DirectColor
            ? 0x80000000u : 0u);
    const float cutoff = settings.gaussianCutoffSigma;
    params.gaussianCutoffDistanceSq = cutoff * cutoff;
    params.maxBounces = static_cast<std::uint32_t>(std::max(
        settings.maxBounces, 1));
    params.indirectLightClamp = settings.indirectLightClamp;
    params.transparentBackground = settings.transparentBackground ? 1u : 0u;
    params.gaussianOverdrawEnabled = rendersProxyOverdraw(settings) ? 1u : 0u;
    params.gaussianOverdrawMax = static_cast<std::uint32_t>(std::max(
        settings.gaussianProxyOverdrawMax, 1));
    params.aovEnabled = settings.aovEnabled ? 1u : 0u;
}

void VulkanRaytracer::updateLights(const Scene& scene)
{
    gpuDevice->synchronize();
    uploadLights(scene);
}

void VulkanRaytracer::uploadLights(const Scene& scene)
{
    // This routine is called while the renderer is idle by uploadScene.  Keep
    // the replacement immutable so a dispatch cannot observe a partially
    // written light record array.
    std::vector<VulkanLightRecord> records;
    records.reserve(scene.getPointLightCount() + scene.getSpotLightCount()
        + scene.getRectLightCount() + scene.getDirectionalLightCount());

    auto copyVec3 = [](float (&dst)[3], const glm::vec3& src) {
        dst[0] = src.x; dst[1] = src.y; dst[2] = src.z;
    };

    for (uint32_t i = 0; i < scene.getPointLightCount(); ++i)
    {
        const PointLight& source = scene.getPointLightsDevice()[i];
        VulkanLightRecord record{};
        record.type = LightInstance::TypePoint;
        copyVec3(record.position, source.position);
        copyVec3(record.color, source.color);
        record.intensity = source.intensity;
        record.params[0] = source.softRadius;
        record.params[7] = source.selectionWeight();
        records.push_back(record);
    }
    for (uint32_t i = 0; i < scene.getSpotLightCount(); ++i)
    {
        const SpotLight& source = scene.getSpotLightsDevice()[i];
        VulkanLightRecord record{};
        record.type = LightInstance::TypeSpot;
        copyVec3(record.position, source.position);
        copyVec3(record.direction, source.direction);
        copyVec3(record.color, source.color);
        record.intensity = source.intensity;
        record.params[0] = source.softRadius;
        record.params[1] = source.innerConeAngle;
        record.params[2] = source.outerConeAngle;
        record.params[7] = source.selectionWeight();
        records.push_back(record);
    }
    for (uint32_t i = 0; i < scene.getRectLightCount(); ++i)
    {
        const RectLight& source = scene.getRectLightsDevice()[i];
        VulkanLightRecord record{};
        record.type = LightInstance::TypeRect;
        record.flags = source.twoSided != 0 ? 1u : 0u;
        copyVec3(record.position, source.position);
        copyVec3(record.direction, source.direction);
        copyVec3(record.color, source.color);
        copyVec3(record.tangent, source.tangent);
        record.intensity = source.intensity;
        record.params[0] = source.width;
        record.params[1] = source.height;
        record.params[2] = static_cast<float>(source.twoSided);
        record.params[3] = source.barnDoorAngle;
        record.params[4] = source.barnDoorLength;
        record.params[7] = source.selectionWeight();
        records.push_back(record);
    }
    for (uint32_t i = 0; i < scene.getDirectionalLightCount(); ++i)
    {
        const DirectionalLight& source = scene.getDirectionalLightsDevice()[i];
        VulkanLightRecord record{};
        record.type = LightInstance::TypeDirectional;
        copyVec3(record.direction, source.direction);
        copyVec3(record.color, source.color);
        record.intensity = source.intensity;
        record.params[0] = source.softAngle;
        record.params[7] = source.selectionWeight();
        records.push_back(record);
    }

    // Emissive SVM programs become triangle light candidates. Store the
    // world-space triangle directly in the compact record so sampling does
    // not depend on backend-private vertex buffers. Emission is evaluated by
    // SVM at the sampled barycentric point in closest-hit.
    const auto meshInstances = scene.getMeshInstances();
    const auto& materials = scene.getMaterials();
    for (uint32_t instanceIndex = 0; instanceIndex < meshInstances.size(); ++instanceIndex)
    {
        const MeshInstance& instance = *meshInstances[instanceIndex];
        const MeshAsset& mesh = instance.getMeshAsset();
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
            VulkanLightRecord record{};
            record.type = 4u;
            record.flags = instanceIndex;
            record.reserved = primitive;
            copyVec3(record.position, a);
            copyVec3(record.direction, b);
            copyVec3(record.color, c);
            record.intensity = area;
            record.params[7] = area;
            records.push_back(record);
        }
    }

    VulkanLightHeader header{};
    header.count = static_cast<uint32_t>(records.size());
    for (const VulkanLightRecord& record : records)
        header.finiteWeight += std::max(record.params[7], 0.0f);
    std::vector<uint8_t> packed(sizeof(header)
        + records.size() * sizeof(VulkanLightRecord), 0u);
    std::memcpy(packed.data(), &header, sizeof(header));
    if (!records.empty())
        std::memcpy(packed.data() + sizeof(header), records.data(),
            records.size() * sizeof(VulkanLightRecord));
    if (!lightsBuffer || lightsBuffer.size() < packed.size())
    {
        lightsBuffer = gpuDevice->buffer<std::byte>(packed.size());
        params.sceneAddress = lightsBuffer.handle().value;
    }
    gpuDevice->upload(lightsBuffer, std::span<const std::byte>(
        reinterpret_cast<const std::byte*>(packed.data()), packed.size()));
    // See updateDescriptors: sceneAddress is the native packed-light handle.
    params.sceneAddress = lightsBuffer.handle().value;
}

void VulkanRaytracer::uploadEnvironment(const Scene& scene)
{
    // Environment records and images may still be referenced by the previous
    // dispatch. Publish replacements only after that dispatch has retired.
    gpuDevice->synchronize();
    const Environment& environment = scene.getEnvironment();
    const auto& textures = scene.getTextures();
    const bool validTexture = environment.textureIndex >= 0
        && static_cast<std::size_t>(environment.textureIndex) < textures.size();

    // The environment owns its sampled image. Depending on uploadMaterials()
    // made a newly imported HDRI point at a stale/out-of-range material slot.
    // Scalar edits retain these images and update only the small record below.
    if (environment.textureIndex != uploadedEnvironmentTextureIndex)
    {
        gpu::Image<std::byte> replacementImage;
        gpu::Image<std::byte> replacementCdf;
        if (validTexture)
        {
            const Texture& hdri = textures[environment.textureIndex];
            const std::vector<float>& pixels = hdri.getPixels();
            const int width = hdri.getWidth();
            const int height = hdri.getHeight();
            const std::size_t valueCount = static_cast<std::size_t>(width)
                * static_cast<std::size_t>(height) * 4u;
            if (width <= 0 || height <= 0 || pixels.size() < valueCount)
                throw std::runtime_error("invalid HDRI pixel storage");

            replacementImage = gpuDevice->image<std::byte>(width, height,
                gpu::ImageUsage::Sampled, gpu::ImageFormat::Rgba32Float);
            gpuDevice->upload(replacementImage,
                std::span<const std::byte>(std::as_bytes(std::span(pixels))));

            std::vector<float> cdf = Environment::computeCdf(
                pixels.data(), width, height, environment.mapping);
            if (cdf.size() != valueCount)
                throw std::runtime_error("invalid HDRI importance CDF");
            replacementCdf = gpuDevice->image<std::byte>(width, height,
                gpu::ImageUsage::Sampled, gpu::ImageFormat::Rgba32Float);
            gpuDevice->upload(replacementCdf,
                std::span<const std::byte>(std::as_bytes(std::span(cdf))));
        }
        environmentImage = std::move(replacementImage);
        environmentCdfImage = std::move(replacementCdf);
        uploadedEnvironmentTextureIndex = environment.textureIndex;
    }

    const std::uint32_t texture = environmentImage
        ? static_cast<std::uint32_t>(environmentImage.sampled_handle().value)
        : VulkanEnvironmentSnapshot::NoTexture;
    const std::uint32_t cdfTexture = environmentCdfImage
        ? static_cast<std::uint32_t>(environmentCdfImage.sampled_handle().value)
        : VulkanEnvironmentSnapshot::NoTexture;

    const VulkanEnvironmentSnapshot snapshot =
        makeEnvironmentSnapshot(environment, texture, cdfTexture);
    const auto snapshotBytes = std::as_bytes(std::span(&snapshot, 1));
    if (environmentBuffer)
        gpuDevice->upload(environmentBuffer, snapshotBytes);
    else
        environmentBuffer = upload_value(*gpuDevice, snapshot);
    params.environmentBuffer =
        static_cast<std::uint32_t>(environmentBuffer.handle().value);
}

void VulkanRaytracer::uploadMaterials(const Scene& scene,
    const std::vector<uint32_t>& svmWords,
    const std::vector<uint32_t>& svmTextureIndices)
{
    gpuDevice->synchronize();
    // Rebuild the immutable sampled-image table while the device is idle.
    // Material bytecode stores scene texture indices; rewrite that table to
    // descriptor-heap indices before publishing the material buffer.  A
    // failed/missing asset remains a valid material and samples white.
    textureImages.clear();
    const float whitePixel[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    textureImages.emplace_back(gpuDevice->image<std::byte>(1, 1,
        gpu::ImageUsage::Sampled, gpu::ImageFormat::Rgba32Float));
    gpuDevice->upload(textureImages.back(),
        std::span<const std::byte>(std::as_bytes(std::span(whitePixel))));
    const std::size_t textureCount = std::min(
        scene.getTextures().size(), maxSceneTextures);
    std::vector<bool> validTextures(textureCount, false);
    for (std::size_t i = 0; i < textureCount; ++i)
    {
        const Texture& texture = scene.getTextures()[i];
        try {
            if (texture.getWidth() <= 0 || texture.getHeight() <= 0)
                throw std::runtime_error("invalid dimensions");
            // getPixels() exposes linear RGBA floats for all host encodings;
            // uploading one canonical format keeps sRGB conversion identical
            // for byte, half, HDR, and EXR source assets.
            const std::vector<float>& pixels = texture.getPixels();
            textureImages.emplace_back(gpuDevice->image<std::byte>(
                texture.getWidth(), texture.getHeight(), gpu::ImageUsage::Sampled,
                gpu::ImageFormat::Rgba32Float));
            gpuDevice->upload(textureImages.back(),
                std::span<const std::byte>(std::as_bytes(std::span(pixels))));
            validTextures[i] = true;
        } catch (const std::exception& error) {
            LOG_WARN("Vulkan texture upload failed; using white fallback for "
                << texture.getName() << ": " << error.what());
            textureImages.emplace_back(gpuDevice->image<std::byte>(1, 1,
                gpu::ImageUsage::Sampled, gpu::ImageFormat::Rgba32Float));
            gpuDevice->upload(textureImages.back(),
                std::span<const std::byte>(std::as_bytes(std::span(whitePixel))));
        }
    }
    // Each sampled image already carries its own heap slot.
    const auto textureSlot = [this](const std::size_t i) {
        return static_cast<std::uint32_t>(textureImages[i].sampled_handle().value);
    };

    std::vector<uint32_t> gpuTextureIndices(svmTextureIndices.size(), textureSlot(0));
    for (std::size_t slot = 0; slot < svmTextureIndices.size(); ++slot)
    {
        const std::size_t sceneIndex = svmTextureIndices[slot];
        if (sceneIndex < textureCount && validTextures[sceneIndex])
            gpuTextureIndices[slot] = textureSlot(sceneIndex + 1);
    }
    // Keep material records, bytecode, and texture-slot indices in one
    // immutable storage allocation. This preserves the existing two
    // descriptor indices in FrameParams while allowing each Material record
    // to carry offsets into the same packed address space.
    struct GpuMaterialRecord {
        uint32_t bytecodeOffset, bytecodeLength, textureOffset, textureCount;
        uint32_t stackSize, shadowOpaque;
    };
    static_assert(sizeof(GpuMaterialRecord) == 24u);
    std::vector<GpuMaterialRecord> records;
    records.reserve(std::max<size_t>(scene.getMaterials().size(), 1u));
    for (const Material& material : scene.getMaterials())
        records.push_back({material.svmBytecodeOffset, material.svmBytecodeLength,
            material.svmTextureOffset, material.svmTextureCount,
            material.svmStackSize, material.shadowOpaque});
    if (records.empty())
        records.emplace_back();
    const uint32_t recordWordCount = static_cast<uint32_t>(
        (records.size() * sizeof(GpuMaterialRecord) + sizeof(uint32_t) - 1u)
        / sizeof(uint32_t));
    const uint32_t wordBase = (recordWordCount + 3u) & ~3u;
    const uint32_t textureBase = (wordBase
        + static_cast<uint32_t>(svmWords.size()) + 3u) & ~3u;
    const size_t totalWordCount = static_cast<size_t>(textureBase)
        + gpuTextureIndices.size();
    std::vector<uint32_t> packed(std::max<size_t>(totalWordCount, wordBase + 1u), 0u);

    for (GpuMaterialRecord& record : records)
    {
        if (record.bytecodeLength != 0u)
            record.bytecodeOffset += wordBase;
        if (record.textureCount != 0u)
            record.textureOffset += textureBase;
    }
    std::memcpy(packed.data(), records.data(), records.size() * sizeof(GpuMaterialRecord));
    if (!svmWords.empty())
        std::memcpy(packed.data() + wordBase, svmWords.data(),
            svmWords.size() * sizeof(uint32_t));
    if (!gpuTextureIndices.empty())
        std::memcpy(packed.data() + textureBase, gpuTextureIndices.data(),
            gpuTextureIndices.size() * sizeof(uint32_t));

    materialBuffer = upload_bytes(*gpuDevice, packed.data(),
        packed.size() * sizeof(uint32_t));
    // Material records and SVM words are separate immutable allocations. The
    // shader reaches the word stream through the explicit svmWords slot.
    svmWordsBuffer = {};
    params.materialsAddress = materialBuffer.handle().value;
    params.svmWords = static_cast<std::uint32_t>(materialBuffer.handle().value);
    params.accumulationBuffer = static_cast<std::uint32_t>(accumulationBuffer.handle().value);
    uploadLights(scene);
}

void VulkanRaytracer::render(const uint32_t frameIndex, const uint32_t sampleIndex)
{
    params.frameIndex = frameIndex;
    params.sampleIndex = sampleIndex;
    params.width = renderWidth;
    params.height = renderHeight;

    // Inside a gpu::Frame this batches into the frame's command buffer; with
    // no frame open it is submitted on its own, which is the offline path.
    gpuDevice->measure(dispatchTimestamp, [this] {
        pipeline.trace({renderWidth, renderHeight, 1}, params);
    });
}

double VulkanRaytracer::lastDispatchMilliseconds()
{
    return dispatchTimestamp.milliseconds();
}

void VulkanRaytracer::copyColorTo(const gpu::ImageHandle target)
{
    if (!target)
        return;
    gpuDevice->copy(colorImage.handle(), target);
}


std::vector<std::byte> VulkanRaytracer::readColor() {
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

std::vector<gpu::float4> VulkanRaytracer::readBeauty()
{
    std::vector<gpu::float4> result(static_cast<std::size_t>(renderWidth)
        * renderHeight);
    gpuDevice->download(std::as_writable_bytes(std::span(result)), colorImage);
    return result;
}

std::vector<std::uint32_t> VulkanRaytracer::readCryptomatte()
{
    std::vector<std::uint32_t> result(static_cast<std::size_t>(renderWidth)
        * renderHeight);
    std::vector<std::byte> bytes(result.size() * sizeof(result.front()));
    gpuDevice->download(std::span<std::byte>(bytes), cryptomatteImage);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

std::vector<gpu::float4> VulkanRaytracer::readPosition()
{
    std::vector<gpu::float4> result(static_cast<std::size_t>(renderWidth)
        * renderHeight);
    std::vector<std::byte> bytes(result.size() * sizeof(result.front()));
    gpuDevice->download(std::span<std::byte>(bytes), positionImage);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}
