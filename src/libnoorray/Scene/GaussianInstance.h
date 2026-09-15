#pragma once

#include "Scene/Scene.h"
#include "Mesh/Assets/Gaussian.h"
#include "Scene/SceneObject.h"

class GaussianInstance : public SceneObject
{
    friend class Scene;
    GaussianAsset* gaussianAsset{};
public:
    GaussianInstance(Scene& scene, const std::string& name, GaussianAsset* gaussianAsset, const Transform& transf);
    GaussianInstance(const GaussianInstance& other);

    std::unique_ptr<SceneObject> clone() const override;

    GaussianAsset& getGaussianAsset() { return *gaussianAsset; }
    const GaussianAsset& getGaussianAsset() const { return *gaussianAsset; }
    GaussianAsset* getGaussianAssetPtr() const { return gaussianAsset; }
    bool hasGaussianAsset() const { return gaussianAsset != nullptr; }

    void onTransformUpdated() override;
};
