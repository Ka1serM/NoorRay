#include "MeshInstance.h"
#include <utility>
MeshInstance::MeshInstance(Scene& scene, const std::string& name, Mesh* mesh, const Transform& transf)
    : SceneObject(scene, name, transf), mesh(mesh)
{}

MeshInstance::MeshInstance(const MeshInstance& other)
    : SceneObject(other),
      mesh(other.mesh)
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
}
