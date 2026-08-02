#pragma once

#include "Scene/Scene.h"
#include "Geometry/Mesh/Assets/GaussianAsset.h"
#include "Scene/SceneObject.h"

class GaussianInstance : public SceneObject
{
    friend class Scene;
    // Owning: the splat data, easily the largest allocation in a scene, is
    // freed as soon as the last instance referencing it goes away.
    GaussianAssetRef gaussianAsset;
    uint32_t sceneInstanceIndex{~0u};
public:
    GaussianInstance(Scene& scene, const std::string& name, const GaussianAssetRef& gaussianAsset, const Transform& transf);
    GaussianInstance(const GaussianInstance& other);

    std::unique_ptr<SceneObject> clone() const override;
    void accept(SceneObjectVisitor& visitor) override;

    uint32_t getGaussianAssetIndex() const { return gaussianAsset.index(); }
    GaussianAssetHandle getGaussianAssetHandle() const { return gaussianAsset.handle(); }
    const GaussianAssetRef& getGaussianAssetRef() const { return gaussianAsset; }
    GaussianAsset& getGaussianAsset() { return *gaussianAsset.get(); }
    const GaussianAsset& getGaussianAsset() const { return *gaussianAsset.get(); }
    bool hasGaussianAsset() const { return gaussianAsset.isValid(); }

    void onTransformUpdated() override;
};
