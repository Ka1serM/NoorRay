#pragma once

#include "api.h"

#include <pxr/base/tf/hashmap.h>
#include <pxr/imaging/hd/instancer.h>

PXR_NAMESPACE_OPEN_SCOPE

class HDNOORRAY_API HdNoorRayInstancer final : public HdInstancer
{
public:
    HdNoorRayInstancer(HdSceneDelegate*, const SdfPath&);

    void Sync(HdSceneDelegate*, HdRenderParam*, HdDirtyBits*) override;
    VtMatrix4dArray ComputeInstanceTransforms(const SdfPath& prototypeId);

private:
    TfHashMap<TfToken, VtValue, TfToken::HashFunctor> primvars_;
    bool visible_{true};
};

PXR_NAMESPACE_CLOSE_SCOPE
