#pragma once

#include "Scene.h"
#include "Mesh/MeshAsset.h"
#include "SceneObject.h"

class Scene;
class MeshAsset;

class MeshInstance : public SceneObject {
    uint32_t meshIndex;
public:
    MeshInstance(Scene& scene, const std::string& name, uint32_t meshIndex, const Transform& transf);
    MeshInstance(const MeshInstance& other);
    
    bool renderUi() override;

    std::unique_ptr<SceneObject> clone() const override;

    uint32_t getMeshIndex() const { return meshIndex; }
    const MeshAsset& getMeshAsset() const { return scene.getMeshAsset(meshIndex); }
    void onTransformUpdated() override;
};
