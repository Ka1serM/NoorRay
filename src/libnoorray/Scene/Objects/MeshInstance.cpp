#include "MeshInstance.h"
#include <utility>
#include "Scene/SceneObjectVisitor.h"

void MeshInstance::accept(SceneObjectVisitor& visitor)
{
    visitor.visit(static_cast<SceneObject&>(*this));
    visitor.visit(*this);
}

MeshInstance::MeshInstance(Scene& scene, const std::string& name, const MeshAssetRef& meshAsset, const Transform& transf)
    : SceneObject(scene, name, transf), meshAsset(meshAsset)
{}

MeshInstance::MeshInstance(const MeshInstance& other)
    : SceneObject(other),
      meshAsset(other.meshAsset)
{}

std::unique_ptr<SceneObject> MeshInstance::clone() const {
    return std::make_unique<MeshInstance>(*this);
}

void MeshInstance::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    if (!scene)
        return;
    scene->setDirtyFlag(TLAS);
    // A mesh may contribute emissive triangles to the light sampler. Their
    // cached world-space vertices must follow the instance transform too.
    scene->setDirtyFlag(Lights);
    scene->markMeshInstanceTransformDirty(scene->getMeshInstanceIndex(this));
}
