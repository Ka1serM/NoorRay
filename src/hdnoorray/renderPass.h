#pragma once

#include "api.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderPass.h>

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
    // The body of _Execute(), which wraps it so that no exception reaches the
    // host across the Hydra callback boundary.
    void _Render(const HdRenderPassStateSharedPtr& renderPassState);
    static void SetBuffersConverged(
        const HdRenderPassAovBindingVector& bindings, bool converged);

    HdNoorRayRenderParam& renderParam_;
    unsigned int observedSceneVersion_{};
    unsigned int observedRenderSettingsVersion_{};
    unsigned int accumulatedSamples_{};
    unsigned int targetWidth_{};
    unsigned int targetHeight_{};
    bool converged_{};
    bool collectionDirty_{true};
    GfMatrix4d cameraTransform_;
    GfMatrix4d projectionMatrix_;

};

PXR_NAMESPACE_CLOSE_SCOPE
