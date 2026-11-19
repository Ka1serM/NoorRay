#include "NoorRay/Session/Session.h"

#include <algorithm>
#include <cmath>

#include "NoorRay/Core/Log.h"
#include "NoorRay/Format/SceneWriter.h"
#include "NoorRay/Import/SceneImporter.h"
#include "NoorRay/Materials/TextureCache.h"
#include "NoorRay/Shaders/Wavefront/WavefrontRecords.h"

namespace NoorRay::Session {

namespace {

bool isColorSlot(const Scene::TextureSlot slot) {
    return slot == Scene::TextureSlot::BaseColor || slot == Scene::TextureSlot::Emission;
}

bool isTextureUsed(const Scene::Scene& scene, const Scene::TextureHandle texture) {
    if (scene.getEnvironment().texture == texture)
        return true;
    return std::ranges::any_of(scene.getMaterials().getValues(), [&](const Scene::Material& material) {
        return std::ranges::any_of(material.textures, [&](const std::optional<Scene::MaterialTexture>& slot) { return slot && slot->texture == texture; });
    });
}

}

std::expected<void, Core::Error> Session::open(NoorRHI::Device& openedDevice, const std::uint32_t width, const std::uint32_t height,
                                               std::shared_ptr<Render::PipelineCache> pipelines) {
    device = &openedDevice;
    Core::Log::info("the session opens at {}x{}", width, height);
    if (auto opened = gpuScene.open(*device); !opened)
        return opened;
    if (auto opened = Core::check(pickReadback.open(*device, 1)); !opened)
        return opened;
    if (auto opened = renderer.open(*device, width, height, std::move(pipelines)); !opened)
        return opened;
    Core::Log::info("the session is open");
    return {};
}

std::expected<void, Core::Error> Session::importFile(const std::filesystem::path& path) {
    if (auto imported = Import::SceneImporter::import(path, scene); !imported)
        return imported;
    return ensureCamera();
}

std::expected<void, Core::Error> Session::saveFile(const std::filesystem::path& path) const {
    return Format::SceneWriter::write(scene, path);
}

std::expected<void, Core::Error> Session::resize(const std::uint32_t width, const std::uint32_t height) {
    return renderer.resize(width, height);
}

void Session::clear() {
    scene.clear();
    renderer.restartAccumulation();
}

//! A scene without a camera gets one that looks at its bounds from the default view, so that every imported file shows something.
std::expected<void, Core::Error> Session::loadMaterialTexture(const Scene::MaterialHandle material, const Scene::TextureSlot slot,
                                                              const std::filesystem::path& path) {
    const Scene::Material* current = scene.find(material);
    if (!current)
        return Core::fail(Core::ErrorCode::InvalidState, "the material does not exist");
    Materials::TextureCache cache(scene);
    std::expected<Scene::TextureHandle, Core::Error> texture = cache.load(path, isColorSlot(slot) ? Materials::ColorSpace::Srgb : Materials::ColorSpace::NonColor);
    if (!texture)
        return std::unexpected(std::move(texture.error()));
    Scene::Material changed = *current;
    const std::optional<Scene::MaterialTexture> previous = changed.texture(slot);
    Scene::MaterialTexture loaded;
    loaded.texture = *texture;
    if (previous)
        loaded.channel = previous->channel;
    changed.texture(slot) = loaded;
    scene.setMaterial(material, changed);
    if (previous && !isTextureUsed(scene, previous->texture))
        scene.remove(previous->texture);
    return {};
}

void Session::clearMaterialTexture(const Scene::MaterialHandle material, const Scene::TextureSlot slot) {
    const Scene::Material* current = scene.find(material);
    if (!current || !current->texture(slot))
        return;
    Scene::Material changed = *current;
    const Scene::TextureHandle texture = changed.texture(slot)->texture;
    changed.texture(slot).reset();
    scene.setMaterial(material, changed);
    if (!isTextureUsed(scene, texture))
        scene.remove(texture);
}

std::expected<void, Core::Error> Session::ensureCamera() {
    if (scene.findActiveCamera())
        return {};
    if (!scene.getCameras().empty()) {
        scene.setActiveCamera(scene.getCameras().keyAt(0));
        return {};
    }
    const Core::Bounds bounds = scene.computeBounds();
    Scene::Camera camera;
    camera.name = "Camera";
    const bool empty = bounds.min.x > bounds.max.x;
    const Core::float3 center = empty ? Core::float3{0.0f, 0.0f, 0.0f} : (bounds.min + bounds.max) * 0.5f;
    const float radius = empty ? 1.0f : std::max(Core::length(bounds.max - bounds.min) * 0.5f, 1e-3f);
    const float distance = radius / std::sin(0.5f * camera.verticalFov);
    const Core::float3 offset{std::cos(Scene::defaultViewPitch) * std::sin(Scene::defaultViewYaw), std::sin(Scene::defaultViewPitch),
                              std::cos(Scene::defaultViewPitch) * std::cos(Scene::defaultViewYaw)};
    camera.transform = Core::Transform::lookAt(center + offset * distance, center, {0.0f, 1.0f, 0.0f});
    camera.previousTransform = camera.transform;
    Scene::fitClipRange(camera, bounds);
    const Scene::CameraHandle added = scene.addCamera(camera);
    scene.addNode({camera.name, camera.transform, {}, added});
    scene.setActiveCamera(added);
    return {};
}

std::expected<void, Core::Error> Session::record(NoorRHI::CommandList& list) {
    if (!device)
        return Core::fail(Core::ErrorCode::InvalidState, "the session is not open");
    if (auto camera = ensureCamera(); !camera)
        return camera;
    if (auto updated = gpuScene.update(scene); !updated)
        return updated;
    if (renderer.getSettings().shading == Render::Shading::PathTraced)
        if (auto built = gpuScene.buildAccelerationStructures(scene); !built)
            return built;
    return renderer.record(list, gpuScene, *scene.findActiveCamera());
}

void Session::requestPick(NoorRHI::CommandList& list, const std::uint32_t x, const std::uint32_t y) {
    if (pickPending || x >= renderer.getWidth() || y >= renderer.getHeight())
        return;
    list.download(renderer.getTargets().getObjectIds(), pickReadback, std::size_t{y} * renderer.getWidth() + x);
    pickPending = true;
}

std::optional<PickResult> Session::takePick() {
    if (!pickPending || !pickReadback.isReady())
        return std::nullopt;
    pickPending = false;
    const auto values = pickReadback.read();
    if (!values || values->empty())
        return PickResult{};
    const std::uint32_t id = values->front();
    if (id == Shaders::invalidIndex)
        return PickResult{};
    if ((id & Shaders::objectIdGaussian) != 0) {
        const std::uint32_t cloud = id & ~Shaders::objectIdGaussian;
        if (cloud >= scene.getGaussianClouds().size())
            return PickResult{};
        return PickResult{scene.getGaussianClouds().keyAt(cloud)};
    }
    if (id >= scene.getInstances().size())
        return PickResult{};
    return PickResult{scene.getInstances().keyAt(id)};
}

}
