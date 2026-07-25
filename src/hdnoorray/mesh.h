#pragma once

#include "api.h"

#include <pxr/imaging/hd/mesh.h>

#include <cstdint>
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
    // The prim owns its asset and its instances. Everything it holds is given
    // up in Finalize (or when the prim switches between mesh and splat), which
    // is what releases the geometry and splat data from device memory.
    void ReleaseInstances(Scene& scene);
    void ReleaseAll(HdNoorRayRenderParam& param);

    // Cached mapping from expanded vertex index → source point index, used
    // by the position-only update path to avoid re-running BuildTriangleMesh.
    std::vector<uint32_t> vertexToPointIndex_;

    MeshAssetRef meshAsset_;
    GaussianAssetRef gaussianAsset_;
    std::vector<SceneObjectHandle> objects_;
    SdfPath boundMaterialId_;
    std::string splatPath_;
};

PXR_NAMESPACE_CLOSE_SCOPE
