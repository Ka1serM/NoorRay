#pragma once

#include "Scene/Scene.h"
#include "Mesh/GaussianAsset.h"
#include "SceneObject.h"

class GaussianInstance : public SceneObject
{
    uint32_t gaussianAssetIndex;
public:
    GaussianInstance(Scene& scene, const std::string& name, uint32_t gaussianAssetIndex, const Transform& transf);
    GaussianInstance(const GaussianInstance& other);

    bool renderUi() override;
    std::unique_ptr<SceneObject> clone() const override;

    uint32_t getGaussianAssetIndex() const { return gaussianAssetIndex; }
    const GaussianAsset& getGaussianAsset() const { return scene.getGaussianAsset(gaussianAssetIndex); }

    void onTransformUpdated() override;
};
