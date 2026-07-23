#pragma once

#include "light.h"

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayDiskLight final
    : public HdNoorRayAnalyticLight
{
public:
    explicit HdNoorRayDiskLight(const SdfPath& id);

private:
    int NoorRayLightType(HdSceneDelegate*) const override;
    void Configure(
        HdSceneDelegate*, const glm::mat4&, LightInstance&,
        float& intensity) const override;
};

PXR_NAMESPACE_CLOSE_SCOPE
