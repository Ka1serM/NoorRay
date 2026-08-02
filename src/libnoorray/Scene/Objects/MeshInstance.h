#pragma once

#include "Scene/Scene.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/SceneObject.h"

class Scene;
class MeshAsset;

class MeshInstance : public SceneObject {
    // Owning: the mesh asset outlives every instance that draws it, and its
    // geometry and acceleration structure are freed with the last one.
    MeshAssetRef meshAsset;
public:
    MeshInstance(Scene& scene, const std::string& name, const MeshAssetRef& meshAsset, const Transform& transf);
    MeshInstance(const MeshInstance& other);


    std::unique_ptr<SceneObject> clone() const override;
    void accept(SceneObjectVisitor& visitor) override;

    uint32_t getMeshIndex() const { return meshAsset.index(); }
    MeshAssetHandle getMeshHandle() const { return meshAsset.handle(); }
    const MeshAssetRef& getMeshAssetRef() const { return meshAsset; }
    MeshAsset& getMeshAsset() { return *meshAsset.get(); }
    const MeshAsset& getMeshAsset() const { return *meshAsset.get(); }
    bool hasMeshAsset() const { return meshAsset.isValid(); }
    void onTransformUpdated() override;
};
