#include "GaussianInstance.h"
#include "Scene/SceneObjectVisitor.h"

void GaussianInstance::accept(SceneObjectVisitor& visitor)
{
    visitor.visit(static_cast<SceneObject&>(*this));
    visitor.visit(*this);
}

GaussianInstance::GaussianInstance(Scene& scene, const std::string& name, const GaussianAssetRef& gaussianAsset, const Transform& transf)
    : SceneObject(scene, name, transf), gaussianAsset(gaussianAsset) {}

GaussianInstance::GaussianInstance(const GaussianInstance& other)
    : SceneObject(other), gaussianAsset(other.gaussianAsset) {}

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
