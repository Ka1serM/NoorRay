#include "renderParam.h"

#include "Raytracing/Runtime/Raytracer.h"

#include <algorithm>

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayRenderParam::HdNoorRayRenderParam()
{
    session.scene.setMutationBarrier([this] {
        session.raytracer->waitForRender();
    });
}

HdNoorRayRenderParam::~HdNoorRayRenderParam()
{
    session.scene.setMutationBarrier({});
    session.raytracer->waitForRender();
}

void HdNoorRayRenderParam::MarkSceneDirty()
{
    sceneVersion_.fetch_add(1, std::memory_order_relaxed);
}

uint64_t HdNoorRayRenderParam::GetSceneVersion() const
{
    return sceneVersion_.load(std::memory_order_relaxed);
}

void HdNoorRayRenderParam::PublishMaterial(
    const SdfPath& id, const Material& material)
{
    uint32_t materialId = 0;
    const auto existing = materials_.find(id);
    if (existing == materials_.end()) {
        materialId = session.scene.add(material);
        materials_[id] = materialId;
    } else {
        materialId = existing->second;
        session.scene.updateMaterial(materialId, material);
    }
    const auto bindings = materialBindings_.find(id);
    if (bindings != materialBindings_.end()) {
        for (const uint32_t meshIndex : bindings->second)
            if (meshIndex < session.scene.getMeshAssets().size())
                session.scene.getMeshAsset(meshIndex).setMaterialId(
                    0, materialId);
    }
    MarkSceneDirty();
}

void HdNoorRayRenderParam::BindMaterial(
    const SdfPath& id, const uint32_t meshIndex)
{
    if (id.IsEmpty())
        return;
    std::vector<uint32_t>& bindings = materialBindings_[id];
    if (std::ranges::find(bindings, meshIndex) == bindings.end())
        bindings.push_back(meshIndex);
    const auto material = materials_.find(id);
    if (material != materials_.end()
        && meshIndex < session.scene.getMeshAssets().size())
        session.scene.getMeshAsset(meshIndex).setMaterialId(
            0, material->second);
}

void HdNoorRayRenderParam::UnbindMaterial(
    const SdfPath& id, const uint32_t meshIndex)
{
    const auto found = materialBindings_.find(id);
    if (found == materialBindings_.end())
        return;
    std::erase(found->second, meshIndex);
    if (found->second.empty())
        materialBindings_.erase(found);
}

PXR_NAMESPACE_CLOSE_SCOPE
