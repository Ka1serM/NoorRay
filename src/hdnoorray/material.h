#pragma once

#include "api.h"

#include <pxr/imaging/hd/material.h>

#include "Shading/Material.h"

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayMaterial final : public HdMaterial
{
public:
    explicit HdNoorRayMaterial(const SdfPath& id);

    void Sync(HdSceneDelegate*, HdRenderParam*, HdDirtyBits*) override;
    HdDirtyBits GetInitialDirtyBitsMask() const override;

    const Material& GetMaterial() const { return material_; }

private:
    Material material_;
};

PXR_NAMESPACE_CLOSE_SCOPE
