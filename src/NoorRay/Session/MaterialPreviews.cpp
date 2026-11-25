#include "NoorRay/Session/MaterialPreviews.h"

#include <algorithm>
#include <bit>
#include <cmath>

#include "NoorRay/Core/Hash.h"
#include "NoorRay/Import/Primitives.h"
#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Render/Raster/ToneMap.h"

namespace NoorRay::Session {

namespace {

//! Kept small, as the stage itself never changes: every preview has the same sphere, camera and lights.
constexpr float previewFov = 0.5f;
constexpr Core::float3 keyLightPosition{-3.0f, 3.5f, 4.0f};
constexpr float keyLightPower = 2000.0f;
constexpr float keyLightRadius = 1.0f;
constexpr float ambient = 0.3f;
constexpr std::uint32_t previewDepth = 8;
//! Only this much of a texture's data feeds the hash, so asking for a preview stays cheap; the name and size cover the rest.
constexpr std::size_t hashedTextureBytes = 4096;

std::uint64_t mix(std::uint64_t hash, const float value) {
    return Core::combineHash(hash, std::bit_cast<std::uint32_t>(value));
}

std::uint64_t mix(std::uint64_t hash, const Core::float3 value) {
    return mix(mix(mix(hash, value.x), value.y), value.z);
}

std::uint64_t hashTexture(const Scene::Scene& scene, const Scene::MaterialTexture& texture, std::uint64_t hash) {
    hash = Core::combineHash(hash, static_cast<std::uint64_t>(texture.wrap) | static_cast<std::uint64_t>(texture.filter) << 8
                                       | std::uint64_t{texture.uvSet} << 16 | std::uint64_t{texture.channel} << 32);
    hash = mix(mix(hash, texture.transform.row0), texture.transform.row1);
    const Scene::Texture* found = scene.find(texture.texture);
    if (!found)
        return Core::combineHash(hash, 0);
    hash = Core::hashString(found->name, hash);
    hash = Core::combineHash(hash, std::uint64_t{found->width} << 32 | found->height);
    hash = Core::combineHash(hash, static_cast<std::uint64_t>(found->format) << 48 ^ found->data.size());
    return Core::hashBytes(std::span(found->data).first(std::min(found->data.size(), hashedTextureBytes)), hash);
}

}

std::uint64_t MaterialPreviews::hashMaterial(const Scene::Scene& scene, const Scene::Material& material) {
    std::uint64_t hash = Core::hashSeed;
    hash = mix(hash, material.baseColor);
    hash = Core::combineHash(hash, std::uint64_t{material.useVertexColor} | std::uint64_t{material.roughnessIsGlossiness} << 1
                                       | static_cast<std::uint64_t>(material.alphaMode) << 8);
    for (const float value :
         {material.alpha, material.alphaCutoff, material.metallic, material.roughness, material.normalScale, material.ior, material.specularIorLevel,
          material.transmission, material.anisotropic, material.anisotropicRotation, material.coat, material.coatRoughness, material.coatNormalScale,
          material.sheenRoughness, material.emissionStrength, material.thinFilmThickness, material.thinFilmIor, material.dispersion})
        hash = mix(hash, value);
    hash = mix(mix(mix(hash, material.specularTint), material.sheenColor), material.emissionColor);
    for (const std::optional<Scene::MaterialTexture>& texture : material.textures)
        hash = texture ? hashTexture(scene, *texture, Core::combineHash(hash, 1)) : Core::combineHash(hash, 0);
    if (!material.graph)
        return hash;
    hash = Core::combineHash(hash, material.graph->getHash());
    for (const Core::float4& value : material.graphParameters)
        hash = mix(mix(hash, Core::float3{value.x, value.y, value.z}), value.w);
    for (const Scene::MaterialTexture& texture : material.graphTextures)
        hash = hashTexture(scene, texture, hash);
    return hash;
}

std::expected<void, Core::Error> MaterialPreviews::open(NoorRHI::Device& openedDevice, std::shared_ptr<Render::PipelineCache> pipelines,
                                                        const std::uint32_t previewSize, const std::uint32_t samples) {
    device = &openedDevice;
    size = std::max(previewSize, 1u);
    if (auto opened = session.open(*device, size, size, std::move(pipelines)); !opened)
        return opened;
    Scene::Scene& scene = session.getScene();
    Import::Primitives::add(scene, Import::Primitive::Sphere);
    sphereMaterial = scene.getMaterials().keyAt(0);

    const Core::Bounds bounds = scene.computeBounds();
    const float radius = std::max(Core::length(bounds.max - bounds.min) * 0.5f / std::sqrt(3.0f), 1e-3f);
    Scene::Camera camera;
    camera.name = "Preview";
    camera.verticalFov = previewFov;
    constexpr float margin = 1.08f;
    const float sinRadius = std::sin(0.5f * previewFov) / margin;
    silhouette = std::tan(std::asin(sinRadius)) / std::tan(0.5f * previewFov);
    camera.transform = Core::Transform::lookAt({0.0f, 0.0f, margin * radius / std::sin(0.5f * previewFov)}, bounds.center(), {0.0f, 1.0f, 0.0f});
    camera.previousTransform = camera.transform;
    Scene::fitClipRange(camera, bounds);
    scene.setActiveCamera(scene.addCamera(camera));

    Scene::Light key;
    key.name = "Key";
    key.type = Scene::LightType::Sphere;
    key.transform = Core::Transform::lookAt(keyLightPosition * (2.0f * radius), bounds.center(), {0.0f, 1.0f, 0.0f});
    key.power = keyLightPower * radius * radius;
    key.radius = keyLightRadius * radius;
    scene.addLight(key);
    Scene::Environment environment;
    environment.color = {ambient, ambient, ambient};
    environment.visibleToCamera = false;
    scene.setEnvironment(environment);

    Render::RendererSettings settings = Render::RendererSettings::pathTraced();
    settings.primaryVisibility = Render::PrimaryVisibility::Traced;
    settings.reuse = Render::Reuse::Off;
    settings.denoiser = Render::Denoiser::Off;
    settings.accumulate = true;
    settings.maxSamples = std::max(samples, 1u);
    settings.wavefront.maxDepth = previewDepth;
    session.getRenderer().setSettings(settings);
    return {};
}

const NoorRHI::Image* MaterialPreviews::request(const Scene::Scene& source, const Scene::MaterialHandle material) {
    const Scene::Material* found = source.find(material);
    if (!found || !device)
        return nullptr;
    const std::uint64_t key = hashMaterial(source, *found);
    if (const auto entry = cache.find(key); entry != cache.end()) {
        entry->second.lastUse = ++useCounter;
        return &entry->second.image;
    }
    if ((!job || job->key != key) && std::ranges::none_of(queue, [&](const auto& queued) { return queued.first == key; }))
        queue.emplace_back(key, material);
    return nullptr;
}

std::expected<bool, Core::Error> MaterialPreviews::record(NoorRHI::CommandList& list, const Scene::Scene& source) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the material previews are not open");
    if (!job)
        if (auto begun = begin(source); !begun)
            return std::unexpected(std::move(begun.error()));
    if (!job)
        return false;
    if (auto recorded = session.record(list); !recorded)
        return std::unexpected(std::move(recorded.error()));
    if (session.getRenderer().isComplete())
        if (auto finished = finish(list); !finished)
            return std::unexpected(std::move(finished.error()));
    return isBusy();
}

//! Takes the next queued material that still exists, still looks the same and has no preview yet, and puts it on the sphere with copies of its textures.
std::expected<void, Core::Error> MaterialPreviews::begin(const Scene::Scene& source) {
    while (!queue.empty()) {
        const auto [key, handle] = queue.front();
        queue.pop_front();
        const Scene::Material* material = source.find(handle);
        if (!material || cache.contains(key) || hashMaterial(source, *material) != key)
            continue;
        Scene::Scene& scene = session.getScene();
        Scene::Material shown = *material;
        Job started{.key = key, .material = handle};
        for (std::optional<Scene::MaterialTexture>& slot : shown.textures) {
            if (!slot)
                continue;
            const Scene::Texture* texture = source.find(slot->texture);
            if (!texture) {
                slot.reset();
                continue;
            }
            slot->texture = scene.addTexture(*texture);
            started.textures.push_back(slot->texture);
        }
        for (Scene::MaterialTexture& slot : shown.graphTextures) {
            const Scene::Texture* texture = source.find(slot.texture);
            if (!texture) {
                shown.graph.reset();
                break;
            }
            slot.texture = scene.addTexture(*texture);
            started.textures.push_back(slot.texture);
        }
        if (!shown.graph) {
            shown.graphParameters.clear();
            shown.graphTextures.clear();
        }
        scene.setMaterial(sphereMaterial, shown);
        session.getRenderer().restartAccumulation();
        job = std::move(started);
        return {};
    }
    return {};
}

//! Keeps the finished image under its key and drops the copied textures.
std::expected<void, Core::Error> MaterialPreviews::finish(NoorRHI::CommandList& list) {
    const NoorRHI::ImageDesc desc{
        .width = size,
        .height = size,
        .format = Render::ToneMap::displayFormat,
        .usage = NoorRHI::ImageUsage::Sampled,
        .name = "NoorRay material preview",
    };
    Entry entry;
    if (auto opened = Core::check(entry.image.open(*device, desc)); !opened)
        return opened;
    list.copy(session.getRenderer().getDisplay(), entry.image);
    if (finished)
        finished(job->material, list, session.getRenderer());
    entry.lastUse = ++useCounter;
    cache.insert_or_assign(job->key, std::move(entry));
    for (const Scene::TextureHandle texture : job->textures)
        session.getScene().remove(texture);
    job.reset();
    ++generation;
    evict();
    return {};
}

//! Drops the least recently shown previews beyond the capacity; a view that still shows one keeps its own reference to the image.
void MaterialPreviews::evict() {
    while (cache.size() > defaultCapacity) {
        const auto oldest = std::ranges::min_element(cache, {}, [](const auto& entry) { return entry.second.lastUse; });
        cache.erase(oldest);
    }
}

}
