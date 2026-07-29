#pragma once

#include "light.h"

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayDomeLight final : public HdNoorRayLight
{
public:
    explicit HdNoorRayDomeLight(const SdfPath& id);

    void Sync(
        HdSceneDelegate*, HdRenderParam*, HdDirtyBits*) override;
    void Finalize(HdRenderParam*) override;

private:
    bool active_{};
};

PXR_NAMESPACE_CLOSE_SCOPE
