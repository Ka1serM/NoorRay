#include "Raytracer.h"

#include "Materials/MaterialX/SlangMaterialCompiler.h"


#include <numeric>
#include <utility>
#include <unordered_map>


#include <noorrhi/interop.hpp>

#include <algorithm>
#include <chrono>
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
#include "Environment/Environment.h"
#include "Camera/CameraInstance.h"
#include "Camera/FisheyeCamera.h"
#include "Camera/RealisticCamera.h"
#include "Camera/ThinLensCamera.h"

namespace
{
constexpr float LightPi = 3.14159265358979323846f;

// Image allocations are whole multiples of this many pixels per side.
constexpr uint32_t AllocationStep = 64u;
// A settled size keeps an allocation up to this many times the area it would
// allocate itself, and shrinks it only once it has held this long.
constexpr uint64_t MaxAllocatedAreaFactor = 2u;
constexpr auto ShrinkDelay = std::chrono::seconds(1);

// The allocation for a logical size: 150% of it per side, so a viewport
// resized within half again its size never reallocates.
uint32_t allocationExtent(const uint32_t logical)
{
    const uint32_t extent = logical + logical / 2u;
    return (extent + AllocationStep - 1u) / AllocationStep * AllocationStep;
}

float lightSelectionLuminance(const glm::vec3 color)
{
    return fmaxf(glm::dot(color, glm::vec3(0.2126f, 0.7152f, 0.0722f)), 0.0f);
}

// The intensity of an inverse square light that delivers about as much. An
// exponent falloff light gives its intensity unattenuated at the centre and
// fades to zero at its radius, like an inverse square light of intensity
// intensity * d^2 at half the radius.
float falloffEquivalentIntensity(const LightFalloff& falloff, const float intensity)
{
    if (falloff.inverseSquared != 0u || falloff.invRadius <= 0.0f)
        return intensity;
    return intensity * 0.25f / (falloff.invRadius * falloff.invRadius);
}

float pointLightSelectionWeight(const PointLight& light)
{
    return lightSelectionLuminance(light.color)
        * fmaxf(falloffEquivalentIntensity(light.falloff, light.intensity), 0.0f) * (4.0f * LightPi);
}

float spotLightSelectionWeight(const SpotLight& light)
{
    const float cone = 2.0f * LightPi * (1.0f - cosf(
        fmaxf(light.innerConeAngle, light.outerConeAngle)
            * LightPi / 180.0f));
    return lightSelectionLuminance(light.color)
        * fmaxf(falloffEquivalentIntensity(light.falloff, light.intensity), 0.0f) * cone;
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

// USpotLightComponent's solid angle for its Lumens conversion.
float spotSolidAngle(const SpotLight& light)
{
    const float inner = std::clamp(light.innerConeAngle, 0.0f, 89.0f) * LightPi / 180.0f;
    const float outer = std::clamp(light.outerConeAngle * LightPi / 180.0f, inner + 0.001f,
        89.0f * LightPi / 180.0f + 0.001f);
    return 2.0f * LightPi * (1.0f - std::cos(outer));
}

// Unreal's local light brightness (UPointLightComponent and
// USpotLightComponent::ComputeLightBrightness) in candela. Unreal works in
// centimetres. NoorRay now uses centimetres too, so photometric intensities
// carry Unreal's 100 * 100 inverse-square scale. Exponent falloff lights ignore their units: their falloff
// replaces the inverse square law, and their intensity is used as it is.
float localLightIntensity(const LightFalloff& falloff, const uint32_t units, const float intensity,
    const float radius, const float length, const float solidAngle)
{
    if (falloff.inverseSquared == 0u)
        return intensity;
    switch (units) {
    case LightUnitsCandelas: return intensity * (100.0f * 100.0f);
    case LightUnitsUnitless: return intensity * 16.0f;
    case LightUnitsLumens: return intensity * (100.0f * 100.0f) / solidAngle;
    case LightUnitsEV: return std::exp2(intensity) * (100.0f * 100.0f);
    case LightUnitsNits: return intensity * 4.0f * LightPi * radius * (radius + 0.5f * length);
    }
    throw std::invalid_argument("unknown light units " + std::to_string(units));
}

// URectLightComponent::ComputeLightBrightness in nits: FRectLightSceneProxy
// emits the brightness over half the rect's area.
float rectLightIntensity(const RectLight& light)
{
    const float halfArea = 0.5f * light.width * light.height;
    if (halfArea <= 0.0f)
        return 0.0f;
    switch (light.units) {
    case LightUnitsUnitless: return light.intensity * 16.0f / halfArea;
    case LightUnitsCandelas: return light.intensity * (100.0f * 100.0f) / halfArea;
    case LightUnitsLumens: return light.intensity * (100.0f * 100.0f) / LightPi / halfArea;
    case LightUnitsEV: return std::exp2(light.intensity) * (100.0f * 100.0f) / halfArea;
    case LightUnitsNits: return 2.0f * light.intensity;
    }
    throw std::invalid_argument("unknown light units " + std::to_string(light.units));
}

// A scene light record as the renderer reads it: its intensity in renderer
// units and its weight for light selection.
PointLight publishedRecord(PointLight record)
{
    record.intensity = localLightIntensity(record.falloff, record.units, record.intensity,
        record.softRadius, record.sourceLength, 4.0f * LightPi);
    record.selectionWeight = pointLightSelectionWeight(record);
    return record;
}

SpotLight publishedRecord(SpotLight record)
{
    record.intensity = localLightIntensity(record.falloff, record.units, record.intensity,
        record.softRadius, record.sourceLength, spotSolidAngle(record));
    record.selectionWeight = spotLightSelectionWeight(record);
    return record;
}

RectLight publishedRecord(RectLight record)
{
    record.intensity = rectLightIntensity(record);
    record.twoSided = record.twoSided != 0 ? 1u : 0u;
    record.selectionWeight = rectLightSelectionWeight(record);
    return record;
}

// Unreal directional light intensity is in lux. Keep its ELightUnits value
// intact in the record, as for every other light type.
DirectionalLight publishedRecord(DirectionalLight record)
{
    record.selectionWeight = directionalLightSelectionWeight(record);
    return record;
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


// Keeps `buffer` at least `count` long, doubling so a growing table is
// replaced O(log n) times. Returns whether it was replaced, which drops its
// contents.
template<class T>
bool reserveTable(noorrhi::Device& device, noorrhi::Buffer<T>& buffer, const std::size_t count)
{
    if (buffer && buffer.size() >= count)
        return false;
    buffer = device.buffer<T>(std::max<std::size_t>({count, buffer ? buffer.size() * 2 : 0, 64}));
    return true;
}

// Uploads the entries of `data` at the sorted, unique `indices`, one upload
// per run of neighbours. Past a few dozen runs a single covering upload is
// cheaper than that many submissions.
template<class T>
void uploadEntries(noorrhi::Buffer<T>& buffer, const std::vector<T>& data,
    const std::vector<uint32_t>& indices)
{
    constexpr std::size_t MaxRuns = 32;
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    for (const uint32_t index : indices) {
        if (!runs.empty() && index == runs.back().second + 1)
            runs.back().second = index;
        else
            runs.emplace_back(index, index);
    }
    if (runs.size() > MaxRuns)
        runs = {{runs.front().first, runs.back().second}};
    for (const auto [first, last] : runs)
        buffer.upload(std::span<const T>(data.data() + first, last - first + 1), first);
}

template<class T>
void uploadTable(noorrhi::Buffer<T>& buffer, const std::vector<T>& data, const std::size_t first = 0)
{
    if (first < data.size())
        buffer.upload(std::span<const T>(data.data() + first, data.size() - first), first);
}

// A TLAS record for an object-to-world transform. `flags` are NoorRHI's
// InstanceFlag* bits.
noorrhi::InstanceRecord instanceRecord(const glm::mat4& objectToWorld, const uint32_t customIndex,
    const uint8_t mask, const uint32_t flags, const uint32_t hitRecordOffset,
    const noorrhi::AccelerationStructure& blas)
{
    noorrhi::InstanceRecord record{};
    // The record is row-major; GLM is column-major.
    for (uint32_t row = 0; row < 3; ++row)
        for (uint32_t column = 0; column < 4; ++column)
            record.transform[row * 4u + column] = objectToWorld[column][row];
    record.custom_index_and_mask = customIndex | (static_cast<uint32_t>(mask) << 24);
    record.shader_binding_table_offset_and_flags = hitRecordOffset | (flags << 24);
    record.blas_address = blas ? blas.handle().value : 0;
    return record;
}

}

RaytracerResources::RaytracerResources(noorrhi::Device& device,
    const uint32_t width, const uint32_t height, const bool exportColorMemory,
    const FullOutputAovs fullOutputAovs, const bool allocateAccumulationBuffer)
    : renderWidth(std::max(width, 1u))
    , renderHeight(std::max(height, 1u))
    , imageWidth_(allocationExtent(renderWidth))
    , imageHeight_(allocationExtent(renderHeight))
    , gpuDevice(&device)
    , splineMeshPass_(device)
    , exportColorMemory(exportColorMemory)
    , allocateAccumulationBuffer(allocateAccumulationBuffer)
    , fullOutputAovs_(fullOutputAovs)
{
    dispatchTimestamp = gpuDevice->timestamp();
    NR_LOG_INFO("graphics API raytracer: using " << gpuDevice->info().name
        << " (rayQuery=" << gpuDevice->features().ray_query << ", rayTracing="
        << gpuDevice->features().ray_tracing << ")");
    NR_LOG_INFO("graphics API raytracer: creating output images");
    createImages();
    NR_LOG_INFO("graphics API raytracer: creating upload buffers");
    lens = noorrhi::Shared<nr::graphics::Lens>(*gpuDevice);

    // An empty pointer table until the first material is published.
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
    NR_LOG_INFO("graphics API raytracer: updating descriptors");
    updateRoot();
    commit();
    NR_LOG_INFO("graphics API raytracer: ready");
}

RaytracerResources::~RaytracerResources() = default;

void RaytracerResources::createImages()
{
    const auto storage = noorrhi::ImageUsage::Storage | noorrhi::ImageUsage::Sampled;
    // Beauty is the authoritative scene-linear HDR image.  Presentation to an
    // 8-bit swapchain is a graphics API blit; offline integrations retain all
    // radiance and alpha values through readBeauty().
    const auto color_usage = exportColorMemory
        ? storage | noorrhi::ImageUsage::ExternalMemory : storage;
    colorImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, color_usage,
        noorrhi::ImageFormat::Rgba32Float);
    cryptomatteImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
        noorrhi::ImageFormat::R32Uint);
    if (fullOutputAovs_ == FullOutputAovs::Written) {
        albedoImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
            noorrhi::ImageFormat::Rgba32Float);
        normalImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
            noorrhi::ImageFormat::Rgba32Float);
        positionImage = gpuDevice->image<std::byte>(imageWidth_, imageHeight_, storage,
            noorrhi::ImageFormat::Rgba32Float);
    }
    // Per-pixel buffers are indexed with the logical width as the stride, so
    // any logical size up to the image allocation fits inside them.
    const std::size_t pixelCount = static_cast<std::size_t>(imageWidth_) * imageHeight_;
    gaussianOverdrawBuffer = gpuDevice->buffer<std::uint32_t>(pixelCount);
    if (allocateAccumulationBuffer)
        accumulationBuffer = gpuDevice->buffer<noorrhi::float4>(pixelCount);
    else
        accumulationBuffer = {};
}

void RaytracerResources::updateRoot()
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

void RaytracerResources::resize(const uint32_t width, const uint32_t height)
{
    if (width == 0 || height == 0
        || (width == renderWidth && height == renderHeight))
        return;
    renderWidth = width;
    renderHeight = height;
    resized_ = std::chrono::steady_clock::now();
    updateRoot();
}

bool RaytracerResources::prepareFrameResources()
{
    if (renderWidth > imageWidth_ || renderHeight > imageHeight_)
    {
        const auto grown = [](const uint32_t allocated, const uint32_t logical) {
            return logical <= allocated ? allocated : allocationExtent(logical);
        };
        reallocate(grown(imageWidth_, renderWidth), grown(imageHeight_, renderHeight));
        return true;
    }
    // A settled size gives back an allocation it leaves mostly unused, such
    // as the one a maximized viewport left behind.
    const uint32_t width = allocationExtent(renderWidth);
    const uint32_t height = allocationExtent(renderHeight);
    if (static_cast<uint64_t>(imageWidth_) * imageHeight_
            <= MaxAllocatedAreaFactor * static_cast<uint64_t>(width) * height
        || std::chrono::steady_clock::now() - resized_ < ShrinkDelay)
        return false;
    reallocate(width, height);
    return true;
}

void RaytracerResources::reallocate(const uint32_t width, const uint32_t height)
{
    // In-flight command buffers may still reference the images being replaced.
    gpuDevice->synchronize();
    imageWidth_ = width;
    imageHeight_ = height;
    createImages();
    if (callbacks_.imageAllocationChanged)
        callbacks_.imageAllocationChanged();
    updateRoot();
}

void RaytracerResources::commit()
{
    lens.commit();
    // The Environment commits its own record; data.environment is published
    // by uploadEnvironment().
    data.lens = lens.ptr().address;
}

void RaytracerResources::uploadScene(Scene& scene)
{
    gpuDevice->synchronize();
    // Everything is republished, so whatever the change lists hold is moot.
    scene.takeChangedMeshInstanceSlots();
    scene.takeChangedMeshes();
    scene.takeChangedMaterials();
    publishedClearEpoch_ = scene.getClearEpoch();
    uploadedTextureCount_ = 0;
    sourceMeshes_.clear();
    sourceMeshIndices_.clear();
    meshRecordData_.clear();
    bindings_.clear();
    bindingIndices_.clear();
    materialBindings_.clear();
    meshBindings_.clear();
    hitRecords_.clear();
    hitRecordsChanged_ = true;
    bindingsMoved_ = false;
    materialPointerData_.clear();
    instanceData_.clear();
    tlasRecordData_.clear();
    slotRecordOffsets_.clear();
    slotColors_.clear();
    slotCustomData_.clear();
    meshRecords_ = {};
    instances_ = {};
    tlasRecords_ = {};
    tlas = {};
    tlasInstanceCount_ = 0;

    uploadTextures(scene);
    std::vector<uint32_t> allMaterials(scene.getMaterials().size());
    std::iota(allMaterials.begin(), allMaterials.end(), 0u);
    publishMaterials(scene, allMaterials);
    for (Mesh& mesh : scene.getMeshes())
        mesh.upload(*gpuDevice, splineMeshPass_);
    std::vector<uint32_t> allSlots(scene.getMeshInstanceSlots().size());
    std::iota(allSlots.begin(), allSlots.end(), 0u);
    publishInstances(scene, std::move(allSlots), true);
    uploadLights(scene);
    applyRenderSettings(scene.getRenderSettings());
}

void RaytracerResources::publishScene(Scene& scene)
{
    if (scene.getClearEpoch() != publishedClearEpoch_) {
        uploadScene(scene);
        return;
    }
    uploadTextures(scene);
    publishMaterials(scene, scene.takeChangedMaterials());
    for (Mesh* mesh : scene.takeChangedMeshes())
        publishMesh(scene, *mesh);
    publishInstances(scene, scene.takeChangedMeshInstanceSlots(), false);
    applyRenderSettings(scene.getRenderSettings());
}

void RaytracerResources::applyRenderSettings(const RenderSettings& settings)
{
    // The SH coefficients splats are shaded with, and above them the flag
    // for the direct-colour shading mode.
    data.gaussianShCoefficientCount = sphericalHarmonicsCoefficientCount(
            settings.gaussianRenderSphericalHarmonics)
        | (settings.gaussianShadingMode == GaussianShadingMode::DirectColor
            ? 0x80000000u : 0u);
    data.maxBounces = static_cast<std::uint32_t>(std::max(
        settings.maxBounces, 1));
    data.indirectLightClamp = settings.indirectLightClamp;
    data.transparentBackground = settings.transparentBackground ? 1u : 0u;
    data.gaussianOverdrawEnabled = rendersProxyOverdraw(settings) ? 1u : 0u;
    data.gaussianOverdrawMax = static_cast<std::uint32_t>(std::max(
        settings.gaussianProxyOverdrawMax, 1));
    if (callbacks_.renderSettingsApplied)
        callbacks_.renderSettingsApplied(settings);
}

void RaytracerResources::updateCamera(const Scene& scene)
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

void RaytracerResources::updateLights(const Scene& scene, const Scene::LightIndices& changed)
{
    if (!lightBuffersValid_
        || scene.getPointLightCount() != pointLightData_.size()
        || scene.getSpotLightCount() != spotLightData_.size()
        || scene.getRectLightCount() != rectLightData_.size()
        || scene.getDirectionalLightCount() != directionalLightData_.size()) {
        uploadLights(scene);
        return;
    }
    // Hiding or showing a light leaves its selection weight alone, so it
    // patches the record without touching the light samplers' tables.
    bool samplingChanged = false;
    const auto patch = [this, &samplingChanged](auto& buffer, auto& mirror, const auto* records,
        std::vector<uint32_t> indices) {
        for (const uint32_t index : indices) {
            const auto record = publishedRecord(records[index]);
            auto shown = record;
            shown.controls.visible = mirror[index].controls.visible;
            samplingChanged |= std::memcmp(&shown, &mirror[index], sizeof(shown)) != 0;
            mirror[index] = record;
        }
        std::ranges::sort(indices);
        uploadEntries(buffer, mirror, indices);
    };
    patch(pointLights, pointLightData_, scene.getPointLights(), changed[LightInstance::TypePoint]);
    patch(spotLights, spotLightData_, scene.getSpotLights(), changed[LightInstance::TypeSpot]);
    patch(rectLights, rectLightData_, scene.getRectLights(), changed[LightInstance::TypeRect]);
    patch(directionalLights, directionalLightData_, scene.getDirectionalLights(),
        changed[LightInstance::TypeDirectional]);
    if (samplingChanged)
        publishLightSampling();
}

void RaytracerResources::uploadLights(const Scene& scene)
{
    // This routine is called while the renderer is idle by uploadScene. Keep
    // each replacement immutable so a dispatch cannot observe a partially
    // written typed light buffer.
    std::vector<nr::graphics::PointLight> pointRecords;
    std::vector<nr::graphics::SpotLight> spotRecords;
    std::vector<nr::graphics::RectLight> rectRecords;
    std::vector<nr::graphics::DirectionalLight> directionalRecords;
    // No material reports emission until the spectral renderer runs MaterialX
    // shaders, so there are no emissive triangles to sample.
    const std::vector<nr::graphics::MeshLight> meshRecords;
    pointRecords.reserve(scene.getPointLightCount());
    spotRecords.reserve(scene.getSpotLightCount());
    rectRecords.reserve(scene.getRectLightCount());
    directionalRecords.reserve(scene.getDirectionalLightCount());

    for (uint32_t i = 0; i < scene.getPointLightCount(); ++i)
        pointRecords.push_back(publishedRecord(scene.getPointLights()[i]));
    for (uint32_t i = 0; i < scene.getSpotLightCount(); ++i)
        spotRecords.push_back(publishedRecord(scene.getSpotLights()[i]));
    for (uint32_t i = 0; i < scene.getRectLightCount(); ++i)
        rectRecords.push_back(publishedRecord(scene.getRectLights()[i]));
    for (uint32_t i = 0; i < scene.getDirectionalLightCount(); ++i)
        directionalRecords.push_back(publishedRecord(scene.getDirectionalLights()[i]));

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
    publishLightSampling();
}

void RaytracerResources::publishLightSampling()
{
    float finiteWeight = 0.0f;
    auto accumulateWeight = [&finiteWeight](const auto& records) {
        for (const auto& record : records)
            finiteWeight += std::max(record.selectionWeight, 0.0f);
    };
    accumulateWeight(pointLightData_);
    accumulateWeight(spotLightData_);
    accumulateWeight(rectLightData_);
    accumulateWeight(directionalLightData_);
    accumulateWeight(meshLightData_);
    data.lightFiniteWeight = finiteWeight;
    if (callbacks_.lightsUploaded)
        callbacks_.lightsUploaded();
}

void RaytracerResources::uploadTextures(Scene& scene)
{
    // Textures are only ever appended, so the ones past the cursor are new.
    auto& textures = scene.getTextures();
    for (; uploadedTextureCount_ < textures.size(); ++uploadedTextureCount_)
    {
        Texture& texture = textures[uploadedTextureCount_];
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

void RaytracerResources::uploadEnvironment(Scene& scene)
{
    // The Environment owns its HDRI and CDF images and uploads them when the
    // texture changes. Per-frame scalar edits only republish its small record.
    ::Environment& environment = scene.getEnvironment();
    uploadTextures(scene);
    const int index = environment.getTextureIndex();
    const bool hasTexture = index >= 0 && static_cast<std::size_t>(index) < scene.getTextures().size();
    const std::pair<uint64_t, int> source{scene.getClearEpoch(), hasTexture ? index : -1};
    if (source != environmentImageSource_) {
        environment.uploadImages(*gpuDevice, hasTexture ? &scene.getTextures()[index] : nullptr);
        environmentImageSource_ = source;
    } else {
        environment.uploadRecord(*gpuDevice);
    }
    data.environment = environment.ptr().address;
}

void RaytracerResources::publishMaterials(Scene& scene, const std::vector<uint32_t>& changed)
{
    auto& sceneMaterials = scene.getMaterials();
    const auto resolveTexture = [this, &scene](const std::uint32_t index) -> std::uint32_t {
        if (index < scene.getTextures().size()
            && scene.getTextures()[index])
            return scene.getTextures()[index].sampledHandle().value;
        return whiteTexture.sampled_handle().value;
    };
    const std::size_t shaderCount = materialShaders_.size();
    materialPointerData_.resize(sceneMaterials.size(), 0);
    for (const uint32_t index : changed) {
        Material& material = sceneMaterials[index];
        if (const auto& shader = material.shaderProgram.shader;
            shader && !materialShaderIndices_.contains(shader.get())) {
            const auto create = [this](const std::vector<std::uint32_t>& spirv,
                                    const std::string_view entryPoint) {
                return gpuDevice->create_shader(std::as_bytes(std::span(spirv)), entryPoint);
            };
            materialShaderIndices_.emplace(shader.get(), static_cast<uint32_t>(materialShaders_.size()));
            materialShaderPrograms_.push_back(shader);
            materialShaders_.push_back({create(shader->closestHit, "closestHit"),
                create(shader->anyHit, "anyHit"), create(shader->shadowAnyHit, "shadowAnyHit")});
        }
        if (!material)
            material.upload(*gpuDevice, resolveTexture);
        materialPointerData_[index] = material.ptr().address;
        // The hit groups and opacity of the sections drawn with it follow
        // the program.
        if (index < materialBindings_.size())
            for (const uint32_t binding : materialBindings_[index]) {
                bindingsMoved_ |= updateBindingOpacity(scene, binding);
                writeBindingHitRecords(scene, binding);
            }
    }
    if (materialShaders_.size() != shaderCount)
        if (callbacks_.materialShadersChanged)
            callbacks_.materialShadersChanged(materialShaders_);

    std::vector<uint32_t> written = changed;
    std::ranges::sort(written);
    written.erase(std::unique(written.begin(), written.end()), written.end());
    if (reserveTable(*gpuDevice, materials, materialPointerData_.size()))
        uploadTable(materials, materialPointerData_);
    else
        uploadEntries(materials, materialPointerData_, written);
    data.materials = materials.ptr().address;
}

void RaytracerResources::publishMesh(const Scene& scene, Mesh& mesh)
{
    mesh.upload(*gpuDevice, splineMeshPass_);
    const auto found = meshBindings_.find(&mesh);
    if (found == meshBindings_.end())
        return;
    const std::size_t recordCount = mesh.getSections().size() * nr::graphics::RaytracingRayTypeCount;
    bool resized = false;
    for (const uint32_t binding : found->second) {
        const std::size_t end = binding + 1 < bindings_.size()
            ? bindings_[binding + 1].hitRecordOffset : hitRecords_.size();
        resized |= end - bindings_[binding].hitRecordOffset != recordCount;
        // The geometry is new, so every BLAS built from it is gone.
        bindings_[binding].blas = {};
        updateBindingOpacity(scene, binding);
    }
    if (resized)
        layoutHitRecords(scene);
    else
        for (const uint32_t binding : found->second)
            writeBindingHitRecords(scene, binding);
    bindingsMoved_ = true;
}

uint32_t RaytracerResources::sourceMesh(Mesh& mesh)
{
    const auto [found, inserted] = sourceMeshIndices_.try_emplace(&mesh,
        static_cast<uint32_t>(sourceMeshes_.size()));
    if (!inserted)
        return found->second;
    mesh.upload(*gpuDevice, splineMeshPass_);
    sourceMeshes_.push_back(&mesh);
    meshRecordData_.push_back(mesh ? mesh.ptr().address : 0);
    return found->second;
}

uint32_t RaytracerResources::binding(const Scene& scene, const MeshInstance& instance)
{
    Mesh& mesh = *instance.getMeshPtr();
    std::vector<uint32_t> materials;
    materials.reserve(instance.getMaterials().size());
    for (const Material* material : instance.getMaterials()) {
        const uint32_t index = scene.getMaterialIndex(material);
        if (index == ~0u)
            throw std::invalid_argument("Mesh instance " + instance.getName()
                + " draws with a material outside its scene");
        materials.push_back(index);
    }
    const auto [found, inserted] = bindingIndices_.try_emplace({&mesh, materials},
        static_cast<uint32_t>(bindings_.size()));
    if (!inserted)
        return found->second;
    const uint32_t index = found->second;
    sourceMesh(mesh);
    MaterialBinding& binding = bindings_.emplace_back();
    binding.mesh = &mesh;
    binding.materials = std::move(materials);
    binding.materialTable = gpuDevice->buffer<uint32_t>(std::max<std::size_t>(binding.materials.size(), 1));
    binding.materialTable.upload(std::span<const uint32_t>(binding.materials));
    binding.hitRecordOffset = static_cast<uint32_t>(hitRecords_.size());
    hitRecords_.resize(hitRecords_.size()
        + mesh.getSections().size() * nr::graphics::RaytracingRayTypeCount);
    materialBindings_.resize(std::max(materialBindings_.size(), scene.getMaterials().size()));
    for (const uint32_t material : binding.materials)
        if (materialBindings_[material].empty() || materialBindings_[material].back() != index)
            materialBindings_[material].push_back(index);
    meshBindings_[&mesh].push_back(index);
    updateBindingOpacity(scene, index);
    writeBindingHitRecords(scene, index);
    return index;
}

bool RaytracerResources::updateBindingOpacity(const Scene& scene, const uint32_t index)
{
    MaterialBinding& binding = bindings_[index];
    const auto& sections = binding.mesh->getSections();
    if (binding.materials.size() < binding.mesh->getSlotCount())
        throw std::invalid_argument("Mesh " + binding.mesh->getPath() + " draws "
            + std::to_string(binding.mesh->getSlotCount()) + " material slots, but an instance supplies "
            + std::to_string(binding.materials.size()));
    const auto& materials = scene.getMaterials();
    const bool castsShadow = bindingCastsShadow(scene, binding);
    std::vector<bool> opacity;
    opacity.reserve(sections.size());
    bool doubleSided = false;
    for (const MeshSection& section : sections)
        doubleSided |= (materials[binding.materials[section.slot]].getData().flags
            & nr::graphics::MaterialFlagOneSided) == 0;
    for (const MeshSection& section : sections) {
        const Material& material = materials[binding.materials[section.slot]];
        // TLAS facing flags cover the whole instance. Mixed-sided meshes
        // need any-hit culling for their one-sided sections.
        const bool needsFacingTest = doubleSided
            && (material.getData().flags & nr::graphics::MaterialFlagOneSided) != 0;
        // Sections that cast no shadow among ones that do reach their shadow
        // any-hit stage, which ignores them.
        opacity.push_back(!material.shaderProgram.transparent && !needsFacingTest
            && (sectionCastsShadow(scene, binding, section) || !castsShadow));
    }
    // Gaussian splat proxies take their own mask, which only the renderers
    // that draw splats trace against.
    const bool splat = !sections.empty()
        && materials[binding.materials[sections.front().slot]].kind == MaterialKind::GaussianSplat;
    const auto mask = static_cast<uint8_t>(splat ? nr::graphics::RaytracingMaskGaussian
        : castsShadow ? nr::graphics::RaytracingMaskMesh
        : nr::graphics::RaytracingMaskMesh & ~nr::graphics::RaytracingMaskShadow);
    const bool facingChanged = binding.doubleSided != doubleSided;
    binding.doubleSided = doubleSided;
    if (binding.blas && opacity == binding.opacity && mask == binding.mask && !facingChanged)
        return false;
    binding.opacity = std::move(opacity);
    binding.mask = mask;
    binding.blas = binding.mesh->blas(*gpuDevice, binding.opacity);
    return true;
}

bool RaytracerResources::sectionCastsShadow(const Scene& scene, const MaterialBinding& binding,
    const MeshSection& section)
{
    return section.castsShadow && (scene.getMaterials()[binding.materials[section.slot]].getData().flags
        & nr::graphics::MaterialFlagNoShadows) == 0;
}

bool RaytracerResources::bindingCastsShadow(const Scene& scene, const MaterialBinding& binding)
{
    return std::ranges::any_of(binding.mesh->getSections(), [&](const MeshSection& section) {
        return sectionCastsShadow(scene, binding, section);
    });
}

void RaytracerResources::writeBindingHitRecords(const Scene& scene, const uint32_t index)
{
    const MaterialBinding& binding = bindings_[index];
    const bool castsShadow = bindingCastsShadow(scene, binding);
    uint32_t record = binding.hitRecordOffset;
    for (const MeshSection& section : binding.mesh->getSections()) {
        const Material& material = scene.getMaterials()[binding.materials[section.slot]];
        const MaterialShaderProgram& program = material.shaderProgram;
        const auto found = program.shader
            ? materialShaderIndices_.find(program.shader.get()) : materialShaderIndices_.end();
        const uint32_t shaders = found == materialShaderIndices_.end() ? ~0u : found->second;
        const HitRecord::Kind kind = material.kind == MaterialKind::GaussianSplat
            ? HitRecord::Kind::Gaussian : HitRecord::Kind::Section;
        const bool shadowFiltered = castsShadow && !sectionCastsShadow(scene, binding, section);
        for (uint32_t rayType = 0; rayType < nr::graphics::RaytracingRayTypeCount; ++rayType, ++record) {
            const HitRecord value{kind, rayType, shaders, program.transparent, shadowFiltered};
            if (hitRecords_[record] == value)
                continue;
            hitRecords_[record] = value;
            hitRecordsChanged_ = true;
        }
    }
}

void RaytracerResources::layoutHitRecords(const Scene& scene)
{
    hitRecords_.clear();
    for (uint32_t index = 0; index < bindings_.size(); ++index) {
        bindings_[index].hitRecordOffset = static_cast<uint32_t>(hitRecords_.size());
        hitRecords_.resize(hitRecords_.size()
            + bindings_[index].mesh->getSections().size() * nr::graphics::RaytracingRayTypeCount);
        writeBindingHitRecords(scene, index);
    }
    hitRecordsChanged_ = true;
    bindingsMoved_ = true;
}

template <class T>
std::uint64_t RaytracerResources::uploadSlotStream(SlotStream<T>& entry, const std::span<const T> data,
    const std::shared_ptr<const void>& owner)
{
    if (data.empty()) {
        entry = {};
        return 0;
    }
    if (!entry.buffer || entry.data != data.data() || entry.buffer.size() != data.size()) {
        entry.owner = owner;
        entry.data = data.data();
        entry.buffer = gpuDevice->buffer<T>(data.size());
        entry.buffer.upload(data);
    }
    return entry.buffer.ptr().address;
}

std::uint64_t RaytracerResources::slotColors(const uint32_t slot, const MeshInstance& instance)
{
    const std::span<const uint32_t> colors = instance.getColors();
    if (!colors.empty() && colors.size() != instance.getMesh().getVertexCount())
        throw std::invalid_argument("Mesh instance " + instance.getName() + " has "
            + std::to_string(colors.size()) + " colors for "
            + std::to_string(instance.getMesh().getVertexCount()) + " vertices");
    return uploadSlotStream(slotColors_[slot], colors, instance.getColorOwner());
}

void RaytracerResources::publishInstances(const Scene& scene, std::vector<uint32_t> changedSlots,
    bool rewriteAll)
{
    const auto& slots = scene.getMeshInstanceSlots();
    const auto count = static_cast<uint32_t>(slots.size());
    const std::size_t sourceMeshesBefore = sourceMeshes_.size();
    const std::size_t bindingsBefore = bindings_.size();
    rewriteAll |= std::exchange(bindingsMoved_, false);
    if (rewriteAll) {
        changedSlots.resize(count);
        std::iota(changedSlots.begin(), changedSlots.end(), 0u);
    }
    std::ranges::sort(changedSlots);

    // A slot whose placement count changed moves the records of every later
    // slot; so do slots that were added or removed.
    const auto previousCount = static_cast<uint32_t>(
        slotRecordOffsets_.empty() ? 0 : slotRecordOffsets_.size() - 1);
    uint32_t firstMoved = std::min(previousCount, count);
    for (const uint32_t slot : changedSlots)
        if (slot < firstMoved && slotRecordOffsets_[slot + 1] - slotRecordOffsets_[slot]
                != slots[slot]->getPlacementCount())
            firstMoved = slot;
    slotRecordOffsets_.resize(count + 1);
    for (uint32_t slot = firstMoved; slot < count; ++slot)
        slotRecordOffsets_[slot + 1] = slotRecordOffsets_[slot] + slots[slot]->getPlacementCount();
    const uint32_t recordCount = slotRecordOffsets_[count];
    std::erase_if(changedSlots, [firstMoved](const uint32_t slot) { return slot >= firstMoved; });
    for (uint32_t slot = firstMoved; slot < count; ++slot)
        changedSlots.push_back(slot);

    instanceData_.resize(count);
    slotColors_.resize(count);
    slotCustomData_.resize(count);
    tlasRecordData_.resize(recordCount);
    std::vector<uint32_t> changedRecords;
    for (const uint32_t slot : changedSlots) {
        const MeshInstance& instance = *slots[slot];
        const MaterialBinding& drawn = bindings_[binding(scene, instance)];
        const std::span<const float> customData = instance.getCustomData();
        instanceData_[slot] = {sourceMeshIndices_.at(drawn.mesh), instance.getMaterialEntry(),
            drawn.materialTable.ptr().address, slotColors(slot, instance),
            uploadSlotStream(slotCustomData_[slot], customData, instance.getCustomDataOwner()),
            instance.getLightingChannels(), static_cast<uint32_t>(customData.size()),
            (instance.getRayTracingFlags().animated || instance.getMesh().isAnimated())
                ? nr::graphics::InstanceFlagAnimated : 0u};
        const MeshInstance::RayTracingFlags rayTracing = instance.getRayTracingFlags();
        // A hidden instance keeps its records, with a mask no ray matches.
        uint8_t mask = instance.isVisible() ? drawn.mask : 0;
        if (mask != nr::graphics::RaytracingMaskGaussian) {
            if (!rayTracing.camera)
                mask &= ~nr::graphics::RaytracingMaskCamera;
            if (!rayTracing.shadow)
                mask &= ~nr::graphics::RaytracingMaskShadow;
            if (!rayTracing.indirect)
                mask &= ~nr::graphics::RaytracingMaskIndirect;
        }
        // Counterclockwise triangles face forward, as they do in Unreal and
        // glTF, unless the instance reverses its culling.
        const uint32_t flags = (drawn.doubleSided ? noorrhi::InstanceFlagTriangleFacingCullDisable : 0u)
            | (rayTracing.reverseCulling ? 0u : noorrhi::InstanceFlagTriangleFrontCounterClockwise);
        const glm::mat4 world = instance.getWorldTransform().getMatrix();
        const std::vector<glm::mat4>& placements = instance.getPlacements();
        for (uint32_t placement = 0; placement < instance.getPlacementCount(); ++placement) {
            const uint32_t record = slotRecordOffsets_[slot] + placement;
            tlasRecordData_[record] = instanceRecord(placements.empty() ? world : world * placements[placement],
                slot, mask, flags, drawn.hitRecordOffset, drawn.blas);
            changedRecords.push_back(record);
        }
    }

    if (reserveTable(*gpuDevice, meshRecords_, meshRecordData_.size()))
        uploadTable(meshRecords_, meshRecordData_);
    else
        uploadTable(meshRecords_, meshRecordData_, sourceMeshesBefore);
    if (reserveTable(*gpuDevice, instances_, count))
        uploadTable(instances_, instanceData_);
    else
        uploadEntries(instances_, instanceData_, changedSlots);
    if (reserveTable(*gpuDevice, tlasRecords_, recordCount))
        uploadTable(tlasRecords_, tlasRecordData_);
    else
        uploadEntries(tlasRecords_, tlasRecordData_, changedRecords);

    if (recordCount == 0) {
        // Publishing an empty scene must retire the previous GPU snapshot.
        // The ray-generation shader treats a null TLAS as an environment miss.
        tlas = {};
    } else if (!tlas || recordCount != tlasInstanceCount_ || rewriteAll
               || bindings_.size() != bindingsBefore) {
        std::vector<noorrhi::AccelerationStructure> referenced;
        referenced.reserve(bindings_.size());
        for (const MaterialBinding& drawn : bindings_)
            if (drawn.blas)
                referenced.push_back(drawn.blas);
        tlas = gpuDevice->build_tlas(tlasRecords_.ptr(), recordCount, referenced);
    } else if (!changedRecords.empty()) {
        gpuDevice->update_tlas(tlas, tlasRecords_.ptr(), recordCount);
    }
    tlasInstanceCount_ = recordCount;

    data.scene = sceneBuffers();
    data.topLevelAS = tlas.handle().value;
    if (std::exchange(hitRecordsChanged_, false))
        if (callbacks_.hitRecordsChanged)
            callbacks_.hitRecordsChanged(hitRecords_);
}

void RaytracerResources::dispatch(const uint32_t frameIndex, const uint32_t sampleIndex,
    const std::function<void()>& record)
{
    // Hosts that skip prepareFrameResources() (the offline path) have no
    // frame open, so waiting for the device here is harmless.
    if (renderWidth > imageWidth_ || renderHeight > imageHeight_)
        prepareFrameResources();
    static const auto clockStart = std::chrono::steady_clock::now();
    data.gameTime = std::chrono::duration<float>(std::chrono::steady_clock::now() - clockStart).count();
    data.frameIndex = frameIndex;
    data.sampleIndex = sampleIndex;
    data.width = renderWidth;
    data.height = renderHeight;

    // Inside a noorrhi::Frame this batches into the frame's command buffer; with
    // no frame open it is submitted on its own, which is the offline path.
    gpuDevice->measure(dispatchTimestamp, [&] { gpuDevice->label("NoorRay Frame", record); });
}

double RaytracerResources::lastDispatchMilliseconds()
{
    return dispatchTimestamp.milliseconds();
}

std::vector<std::byte> RaytracerResources::readColor() {
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
std::vector<T> RaytracerResources::cropToRender(std::vector<T> pixels) const
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

std::vector<noorrhi::float4> RaytracerResources::readBeauty()
{
    std::vector<noorrhi::float4> result(static_cast<std::size_t>(imageWidth_)
        * imageHeight_);
    colorImage.download(std::as_writable_bytes(std::span(result)));
    return cropToRender(std::move(result));
}

std::vector<std::uint32_t> RaytracerResources::readCryptomatte()
{
    std::vector<std::uint32_t> result(static_cast<std::size_t>(imageWidth_)
        * imageHeight_);
    std::vector<std::byte> bytes(result.size() * sizeof(result.front()));
    cryptomatteImage.download(std::span<std::byte>(bytes));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return cropToRender(std::move(result));
}

std::vector<noorrhi::float4> RaytracerResources::readPosition()
{
    std::vector<noorrhi::float4> result(static_cast<std::size_t>(imageWidth_)
        * imageHeight_);
    std::vector<std::byte> bytes(result.size() * sizeof(result.front()));
    positionImage.download(std::span<std::byte>(bytes));
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return cropToRender(std::move(result));
}

std::uint32_t RaytracerResources::readCryptomatteAt(const uint32_t x, const uint32_t y)
{
    std::uint32_t id = ~0u;
    if (x >= renderWidth || y >= renderHeight)
        return id;
    std::array<std::byte, sizeof(id)> bytes{};
    cryptomatteImage.download_region(x, y, 1, 1, std::span<std::byte>(bytes));
    std::memcpy(&id, bytes.data(), bytes.size());
    return id;
}

noorrhi::float4 RaytracerResources::readPositionAt(const uint32_t x, const uint32_t y)
{
    noorrhi::float4 position{};
    if (x >= renderWidth || y >= renderHeight)
        return position;
    std::array<std::byte, sizeof(position)> bytes{};
    positionImage.download_region(x, y, 1, 1, std::span<std::byte>(bytes));
    std::memcpy(&position, bytes.data(), bytes.size());
    return position;
}

nr::graphics::Scene RaytracerResources::sceneBuffers() const
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
