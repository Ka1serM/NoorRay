#pragma once

#include "Scene/Scene.h"
#include "Mesh/GaussianAsset.h"
#include "SceneObject.h"

class GaussianInstance : public SceneObject
{
    friend class Scene;
    uint32_t gaussianAssetIndex;
    uint32_t sceneInstanceIndex{~0u};
public:
    GaussianInstance(Scene& scene, const std::string& name, uint32_t gaussianAssetIndex, const Transform& transf);
    GaussianInstance(const GaussianInstance& other);

    std::unique_ptr<SceneObject> clone() const override;
    void accept(SceneObjectVisitor& visitor) override;

    uint32_t getGaussianAssetIndex() const { return gaussianAssetIndex; }
    const GaussianAsset& getGaussianAsset() const { return scene->getGaussianAsset(gaussianAssetIndex); }

    void onTransformUpdated() override;
};
