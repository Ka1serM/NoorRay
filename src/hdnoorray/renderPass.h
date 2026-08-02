#pragma once

#include "api.h"

#include <cstddef>
#include <cuda_runtime_api.h>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/imaging/hd/aov.h>
#include <pxr/imaging/hd/renderPass.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdNoorRayRenderParam;
class HdNoorRayRenderBuffer;

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
    void _ReleaseViewportPresenter();
    bool _EnsureViewportPresenter(
        unsigned int texture, unsigned int width, unsigned int height);
    bool _QueueViewportImage(
        cudaArray_t source, unsigned int width, unsigned int height,
        cudaStream_t renderStream);
    bool _QueueViewportBuffer(
        const void* source, unsigned int width, unsigned int height,
        cudaStream_t renderStream);
    bool _PresentViewportImage(
        unsigned int texture, unsigned int width, unsigned int height);
    bool _EnsureOfflineStaging(size_t bytes);
    bool _QueueOfflinePresentation(
        cudaArray_t source, unsigned int width, unsigned int height,
        cudaStream_t renderStream);
    bool _QueueOfflinePresentation(
        const void* source, unsigned int width, unsigned int height,
        cudaStream_t renderStream);
    bool _CommitOfflinePresentation(
        HdNoorRayRenderBuffer& colorBuffer, size_t bytes);

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
    bool framePending_{};
    unsigned int pendingSamples_{};
    bool offlinePresentationPending_{};
    void* offlineStaging_{};
    size_t offlineStagingCapacity_{};
    cudaEvent_t offlinePresentationReadyEvent_{};
    cudaArray_t viewportImage_{};
    unsigned int viewportImageWidth_{};
    unsigned int viewportImageHeight_{};
    bool viewportImageValid_{};
    cudaGraphicsResource_t viewportGraphicsResource_{};
    unsigned int viewportGraphicsTexture_{};
    unsigned int viewportGraphicsWidth_{};
    unsigned int viewportGraphicsHeight_{};
    cudaStream_t viewportCopyStream_{};
    cudaEvent_t viewportImageReadyEvent_{};
    bool viewportImageEventRecorded_{};
    cudaEvent_t viewportCopyFinishedEvent_{};
    bool viewportCopyFinishedEventRecorded_{};
    GfMatrix4d cameraTransform_;
    GfMatrix4d projectionMatrix_;

};

PXR_NAMESPACE_CLOSE_SCOPE
