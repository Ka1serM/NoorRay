#pragma once

#include "api.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/imaging/hd/renderPass.h>

#include <cstdint>

PXR_NAMESPACE_OPEN_SCOPE

class HdNoorRayRenderParam;

class HDNOORRAY_API HdNoorRayRenderPass final : public HdRenderPass
{
public:
    HdNoorRayRenderPass(
        HdRenderIndex* index,
        const HdRprimCollection& collection,
        HdNoorRayRenderParam& renderParam);
    ~HdNoorRayRenderPass() override;

    bool IsConverged() const override;

protected:
    void _Execute(
        const HdRenderPassStateSharedPtr& renderPassState,
        const TfTokenVector& renderTags) override;
    void _MarkCollectionDirty() override;

private:
    HdNoorRayRenderParam& renderParam_;
    uint64_t observedSceneVersion_{};
    unsigned int accumulatedSamples_{};
    bool converged_{};
    bool collectionDirty_{true};
    GfMatrix4d cameraTransform_;
    GfMatrix4d projectionMatrix_;
};

PXR_NAMESPACE_CLOSE_SCOPE
