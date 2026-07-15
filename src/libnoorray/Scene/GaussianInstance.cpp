#include "GaussianInstance.h"
#include "Scene/SceneObjectVisitor.h"

void GaussianInstance::accept(SceneObjectVisitor& visitor)
{
    visitor.visit(static_cast<SceneObject&>(*this));
    visitor.visit(*this);
}

GaussianInstance::GaussianInstance(Scene& scene, const std::string& name, const uint32_t gaussianAssetIndex, const Transform& transf)
    : SceneObject(scene, name, transf), gaussianAssetIndex(gaussianAssetIndex) {}

GaussianInstance::GaussianInstance(const GaussianInstance& other)
    : SceneObject(other), gaussianAssetIndex(other.gaussianAssetIndex) {}

std::unique_ptr<SceneObject> GaussianInstance::clone() const
{
    return std::make_unique<GaussianInstance>(*this);
}

void GaussianInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    if (!scene)
        return;
    scene->setDirtyFlag(TLAS);
    scene->markGaussianInstanceTransformDirty(sceneInstanceIndex);
}
