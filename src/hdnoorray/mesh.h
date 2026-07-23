#pragma once

#include "api.h"

#include <pxr/imaging/hd/mesh.h>

#include <cstdint>
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE

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
    uint32_t meshIndex_{~0u};
    std::vector<uint64_t> objectIds_;
    SdfPath boundMaterialId_;
};

PXR_NAMESPACE_CLOSE_SCOPE
