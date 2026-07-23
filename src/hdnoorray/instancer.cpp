#include "instancer.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

PXR_NAMESPACE_OPEN_SCOPE

HdNoorRayInstancer::HdNoorRayInstancer(
    HdSceneDelegate* delegate, const SdfPath& id)
    : HdInstancer(delegate, id)
{
}

void HdNoorRayInstancer::Sync(
    HdSceneDelegate* delegate, HdRenderParam*, HdDirtyBits* dirtyBits)
{
    if (*dirtyBits & HdChangeTracker::DirtyVisibility)
        visible_ = delegate->GetVisible(GetId());
    _UpdateInstancer(delegate, dirtyBits);
    if (HdChangeTracker::IsAnyPrimvarDirty(*dirtyBits, GetId())) {
        for (const HdPrimvarDescriptor& descriptor :
             delegate->GetPrimvarDescriptors(GetId(), HdInterpolationInstance)) {
            if (HdChangeTracker::IsPrimvarDirty(
                    *dirtyBits, GetId(), descriptor.name))
                primvars_[descriptor.name] = delegate->Get(GetId(), descriptor.name);
        }
    }
    *dirtyBits = HdChangeTracker::Clean;
}

VtMatrix4dArray HdNoorRayInstancer::ComputeInstanceTransforms(
    const SdfPath& prototypeId)
{
    if (!visible_)
        return {};
    const VtIntArray indices =
        GetDelegate()->GetInstanceIndices(GetId(), prototypeId);
    VtMatrix4dArray result(indices.size(), GetDelegate()->GetInstancerTransform(GetId()));

    const auto transforms = primvars_.find(HdInstancerTokens->instanceTransforms);
    if (transforms != primvars_.end()
        && transforms->second.IsHolding<VtMatrix4dArray>()) {
        const VtMatrix4dArray& values =
            transforms->second.UncheckedGet<VtMatrix4dArray>();
        for (size_t i = 0; i < indices.size(); ++i)
            if (indices[i] >= 0 && static_cast<size_t>(indices[i]) < values.size())
                result[i] = values[indices[i]] * result[i];
    }
    const auto translations =
        primvars_.find(HdInstancerTokens->instanceTranslations);
    if (translations != primvars_.end()
        && translations->second.IsHolding<VtVec3fArray>()) {
        const VtVec3fArray& values =
            translations->second.UncheckedGet<VtVec3fArray>();
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= values.size())
                continue;
            GfMatrix4d transform(1.0);
            transform.SetTranslate(GfVec3d(values[indices[i]]));
            result[i] = transform * result[i];
        }
    }
    const auto scales = primvars_.find(HdInstancerTokens->instanceScales);
    if (scales != primvars_.end() && scales->second.IsHolding<VtVec3fArray>()) {
        const VtVec3fArray& values = scales->second.UncheckedGet<VtVec3fArray>();
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= values.size())
                continue;
            GfMatrix4d transform(1.0);
            transform.SetScale(GfVec3d(values[indices[i]]));
            result[i] = transform * result[i];
        }
    }
    const auto rotations = primvars_.find(HdInstancerTokens->instanceRotations);
    if (rotations != primvars_.end()
        && rotations->second.IsHolding<VtVec4fArray>()) {
        const VtVec4fArray& values =
            rotations->second.UncheckedGet<VtVec4fArray>();
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || static_cast<size_t>(indices[i]) >= values.size())
                continue;
            const GfVec4f& q = values[indices[i]];
            GfMatrix4d transform(1.0);
            transform.SetRotate(GfQuatd(q[0], q[1], q[2], q[3]));
            result[i] = transform * result[i];
        }
    }

    if (GetParentId().IsEmpty())
        return result;
    auto* parent = dynamic_cast<HdNoorRayInstancer*>(
        GetDelegate()->GetRenderIndex().GetInstancer(GetParentId()));
    if (parent == nullptr)
        return result;
    const VtMatrix4dArray parentTransforms =
        parent->ComputeInstanceTransforms(GetId());
    VtMatrix4dArray flattened(parentTransforms.size() * result.size());
    for (size_t p = 0; p < parentTransforms.size(); ++p)
        for (size_t i = 0; i < result.size(); ++i)
            flattened[p * result.size() + i] = result[i] * parentTransforms[p];
    return flattened;
}

PXR_NAMESPACE_CLOSE_SCOPE
