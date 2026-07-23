#pragma once

#include "api.h"

#include <pxr/imaging/hd/light.h>

#include <cstdint>

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayLight final : public HdLight
{
public:
    HdNoorRayLight(const SdfPath& id, const TfToken& typeId);

    void Sync(HdSceneDelegate*, HdRenderParam*, HdDirtyBits*) override;
    void Finalize(HdRenderParam*) override;
    HdDirtyBits GetInitialDirtyBitsMask() const override;

private:
    TfToken typeId_;
    uint64_t objectId_{};
};

PXR_NAMESPACE_CLOSE_SCOPE
