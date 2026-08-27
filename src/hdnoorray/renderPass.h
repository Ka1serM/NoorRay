#pragma once

#include "api.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/imaging/hd/renderPass.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdNoorRayRenderParam;

class HDNOORRAY_API HdNoorRayRenderPass final : public HdRenderPass
{
public:
    HdNoorRayRenderPass(HdRenderIndex*, const HdRprimCollection&,
        HdNoorRayRenderParam&);
    ~HdNoorRayRenderPass() override;
    bool IsConverged() const override;

protected:
    void _Execute(const HdRenderPassStateSharedPtr&, const TfTokenVector&) override;
    void _MarkCollectionDirty() override;

private:
    void _Render(const HdRenderPassStateSharedPtr&);
    bool _EnsureInteropImage(unsigned int width, unsigned int height);
    bool _PresentLastFrame(unsigned int targetTexture, unsigned int width,
        unsigned int height);
    bool _PresentInterop(unsigned int targetTexture, unsigned int width,
        unsigned int height, int semaphoreFd);
    void _ReleaseInteropImage();
    static void SetBuffersConverged(const HdRenderPassAovBindingVector&, bool);

    HdNoorRayRenderParam& renderParam_;
    unsigned int observedSceneVersion_{};
    unsigned int observedRenderSettingsVersion_{};
    unsigned int accumulatedSamples_{};
    unsigned int targetWidth_{};
    unsigned int targetHeight_{};
    bool converged_{};
    bool collectionDirty_{true};
    unsigned int interopMemory_{};
    unsigned int interopTexture_{};
    unsigned int interopReadFramebuffer_{};
    unsigned int interopWidth_{};
    unsigned int interopHeight_{};
    unsigned int cachedViewportTexture_{};
    unsigned int cachedViewportWidth_{};
    unsigned int cachedViewportHeight_{};
    GfMatrix4d cameraTransform_;
    GfMatrix4d projectionMatrix_;
};

PXR_NAMESPACE_CLOSE_SCOPE
