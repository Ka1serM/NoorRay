#pragma once

#include "api.h"

#include <pxr/imaging/hd/material.h>

#include <vector>

#include "Scene/SceneResources.h"
#include "Shading/Material.h"

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayMaterial final : public HdMaterial
{
public:
    explicit HdNoorRayMaterial(const SdfPath& id);

    void Sync(HdSceneDelegate*, HdRenderParam*, HdDirtyBits*) override;
    void Finalize(HdRenderParam*) override;
    HdDirtyBits GetInitialDirtyBitsMask() const override;

    const Material& GetMaterial() const { return material_; }

private:
    Material material_;
    // The material struct stores bare texture slot indices, so the prim holds
    // the ownership of the textures it samples for as long as it exists.
    std::vector<TextureRef> textures_;
};

PXR_NAMESPACE_CLOSE_SCOPE
