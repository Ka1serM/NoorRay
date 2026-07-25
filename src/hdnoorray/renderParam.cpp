#include "renderParam.h"

#include <pxr/base/tf/diagnostic.h>

#include <algorithm>
#include <exception>

#include "Raytracing/Runtime/Raytracer.h"
#include "Scene/Texture.h"

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayRenderParam::HdNoorRayRenderParam()
{
    // Mutations no longer block on render completion. The GPU stream is
    // synchronized once per frame in Raytracer::renderFrame() via the
    // Scene::consumeGpuSync() mechanism, which batches all changes from a
    // single Hydra commit into one sync point.
    if (session.raytracer)
        session.raytracer->setTimingEnabled(true);
}

HdNoorRayRenderParam::~HdNoorRayRenderParam()
{
    session.scene.setMutationBarrier({});
    if (session.raytracer) {
        session.raytracer->waitForRender();
    }
    // Give up every reference before the session (and with it the registries
    // they point into) is destroyed.
    textureCache_.clear();
    materialBindings_.clear();
    materials_.clear();
    if (session.raytracer)
        session.raytracer.reset();
}

TextureRef HdNoorRayRenderParam::GetOrCreateTexture(
    const std::string& filePath, const TextureEncoding encoding)
{
    PruneTextureCache();
    const auto existing = textureCache_.find(filePath);
    if (existing != textureCache_.end() && existing->second.isValid())
        return existing->second;

    try {
        TextureRef texture = session.scene.add(Texture(filePath, encoding));
        textureCache_[filePath] = texture;
        return texture;
    } catch (const std::exception& error) {
        TF_WARN("hdNoorRay could not load texture '%s': %s",
            filePath.c_str(), error.what());
        return {};
    }
}

void HdNoorRayRenderParam::PruneTextureCache()
{
    // The cache is the only thing keeping a texture alive once no material
    // samples it, so an entry that nothing else references is dropped. Doing it
    // lazily here keeps textures shared across a whole material reassignment.
    std::erase_if(textureCache_, [](const auto& entry) {
        return !entry.second.isValid();
    });
}

void HdNoorRayRenderParam::MarkSceneDirty()
{
    sceneNeedsUpdate_.store(true, std::memory_order_relaxed);
}

void HdNoorRayRenderParam::MarkRenderSettingsChanged()
{
    renderSettingsChanged_.store(true, std::memory_order_relaxed);
}

bool HdNoorRayRenderParam::ConsumeRenderSettingsChanged()
{
    return renderSettingsChanged_.exchange(false, std::memory_order_relaxed);
}

uint64_t HdNoorRayRenderParam::GetSceneVersion()
{
    if (sceneNeedsUpdate_.exchange(false, std::memory_order_relaxed))
        sceneVersion_.fetch_add(1, std::memory_order_relaxed);
    return sceneVersion_.load(std::memory_order_relaxed);
}

void HdNoorRayRenderParam::SetProgress(const double progress)
{
    progress_.store(std::clamp(progress, 0.0, 1.0), std::memory_order_relaxed);
}

double HdNoorRayRenderParam::GetProgress() const
{
    return progress_.load(std::memory_order_relaxed);
}

void HdNoorRayRenderParam::AccumulateGpuTimeMs(const float ms)
{
    cumulativeGpuTimeSeconds_.fetch_add(ms / 1000.0, std::memory_order_relaxed);
}

double HdNoorRayRenderParam::GetTotalClockTime() const
{
    return cumulativeGpuTimeSeconds_.load(std::memory_order_relaxed);
}

void HdNoorRayRenderParam::ResetClock()
{
    cumulativeGpuTimeSeconds_.store(0.0, std::memory_order_relaxed);
    progress_.store(0.0, std::memory_order_relaxed);
}

void HdNoorRayRenderParam::PublishMaterial(
    const SdfPath& id, const Material& material)
{
    MaterialRef& published = materials_[id];
    if (published.isValid())
        session.scene.updateMaterial(published.handle(), material);
    else
        published = session.scene.add(material);

    const auto bindings = materialBindings_.find(id);
    if (bindings != materialBindings_.end())
        for (const MeshAssetHandle mesh : bindings->second)
            if (MeshAsset* asset = session.scene.getMeshAsset(mesh))
                asset->setMaterial(0, published);
    MarkSceneDirty();
}

void HdNoorRayRenderParam::ReleaseMaterial(const SdfPath& id)
{
    materials_.erase(id);
    materialBindings_.erase(id);
    MarkSceneDirty();
}

void HdNoorRayRenderParam::BindMaterial(
    const SdfPath& id, const MeshAssetHandle mesh)
{
    if (id.IsEmpty() || !mesh.isValid())
        return;
    std::vector<MeshAssetHandle>& bindings = materialBindings_[id];
    if (std::ranges::find(bindings, mesh) == bindings.end())
        bindings.push_back(mesh);

    const auto material = materials_.find(id);
    if (material == materials_.end() || !material->second.isValid())
        return;
    if (MeshAsset* asset = session.scene.getMeshAsset(mesh))
        asset->setMaterial(0, material->second);
}

void HdNoorRayRenderParam::UnbindMaterial(
    const SdfPath& id, const MeshAssetHandle mesh)
{
    const auto found = materialBindings_.find(id);
    if (found == materialBindings_.end())
        return;
    std::erase(found->second, mesh);
    if (found->second.empty())
        materialBindings_.erase(found);
}

PXR_NAMESPACE_CLOSE_SCOPE
