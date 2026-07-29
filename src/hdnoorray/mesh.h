#pragma once

#include "api.h"

#include <pxr/imaging/hd/mesh.h>

#include <string>
#include <vector>

#include "Scene/Handle.h"
#include "Scene/SceneResources.h"

class Scene;

PXR_NAMESPACE_OPEN_SCOPE

class HdNoorRayRenderParam;

class HDNOORRAY_API HdNoorRayMesh final : public HdMesh
{
public:
    explicit HdNoorRayMesh(const SdfPath& id);
    ~HdNoorRayMesh() override;

    HdDirtyBits GetInitialDirtyBitsMask() const override;
    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits,
        const TfToken& reprToken) override;
    void Finalize(HdRenderParam* renderParam) override;

protected:
    HdDirtyBits _PropagateDirtyBits(HdDirtyBits bits) const override;
    void _InitRepr(const TfToken& reprToken, HdDirtyBits* dirtyBits) override;

private:
    void ReleaseInstances(Scene& scene);
    void ReleaseAll(HdNoorRayRenderParam& param);
    void UnbindAllMaterials(HdNoorRayRenderParam& param);

    MeshAssetRef meshAsset_;
    GaussianAssetRef gaussianAsset_;
    std::vector<SceneObjectHandle> objects_;
    // One entry per material slot this mesh currently has bound (slot 0 is
    // always the Rprim's own GetMaterialId(); slots 1+ come from HdGeomSubset
    // material bindings, see BuildTriangleMesh's own comment). Index-parallel
    // with MeshAsset::getMaterialIds().
    std::vector<SdfPath> boundMaterialIds_;
    std::string splatPath_;
};

PXR_NAMESPACE_CLOSE_SCOPE
