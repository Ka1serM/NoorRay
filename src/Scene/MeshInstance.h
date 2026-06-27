#pragma once

#include <memory>

#include "Scene.h"
#include "Mesh/MeshAsset.h"
#include "SceneObject.h"

class Scene;
class MeshAsset;

class MeshInstance : public SceneObject {
    const std::shared_ptr<MeshAsset> meshAsset;
public:
    MeshInstance(Scene& scene, const std::string& name, std::shared_ptr<MeshAsset> asset, const Transform& transf);
    MeshInstance(const MeshInstance& other);
    
    void renderUi() override;

    std::unique_ptr<SceneObject> clone() const override;

    const MeshAsset& getMeshAsset() const { return *meshAsset.get(); }
    void onTransformUpdated() override;
};
