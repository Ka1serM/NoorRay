#include "MeshInstance.h"
#include <utility>
#include "Scene/SceneObjectVisitor.h"

void MeshInstance::accept(SceneObjectVisitor& visitor)
{
    visitor.visit(static_cast<SceneObject&>(*this));
    visitor.visit(*this);
}

MeshInstance::MeshInstance(Scene& scene, const std::string& name, const uint32_t meshIndex, const Transform& transf)
    : SceneObject(scene, name, transf), meshIndex(meshIndex)
{}

MeshInstance::MeshInstance(const MeshInstance& other)
    : SceneObject(other),
      meshIndex(other.meshIndex)
{}

std::unique_ptr<SceneObject> MeshInstance::clone() const {
    return std::make_unique<MeshInstance>(*this);
}

void MeshInstance::onTransformUpdated() {
    SceneObject::onTransformUpdated();
    if (!scene)
        return;
    scene->setDirtyFlag(TLAS);
    scene->markMeshInstanceTransformDirty(scene->getMeshInstanceIndex(this));
}
