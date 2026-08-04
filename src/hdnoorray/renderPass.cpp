#include "renderPass.h"

#include "renderBuffer.h"
#include "renderParam.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <vector>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "Rendering/Camera/Camera.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/GatherPsfSensor.h"
#include "Rendering/Camera/RectangularSensor.h"
#include "Rendering/Camera/ScatterPsfSensor.h"
#include "Rendering/Camera/HybridPsfCamera.h"
#include "Rendering/Camera/RealisticCamera.h"
#include "Backend/CUDA/ManagedMemory.h"
#include "IO/Bitmap.h"
#include "Backend/OptiX/Runtime/Raytracer.h"
#include "Scene/Scene.h"
#include "Backend/Vulkan/Runtime/Image.h"
#include "stb_image_resize2.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
glm::mat4 ToGlm(const GfMatrix4d& value)
{
    glm::mat4 result(1.0f);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            // GfMatrix uses row vectors while GLM/NoorRay use column vectors.
            result[column][row] = static_cast<float>(value[column][row]);
    return result;
}

HdNoorRayRenderBuffer* GetBuffer(
    const HdRenderPassAovBinding& binding)
{
    return dynamic_cast<HdNoorRayRenderBuffer*>(binding.renderBuffer);
}

bool GetOpenGlColorTarget(
    GLuint& texture, unsigned int& width, unsigned int& height)
{
    GLint objectType = GL_NONE;
    GLint objectName = 0;
    glGetFramebufferAttachmentParameteriv(
        GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &objectType);
    glGetFramebufferAttachmentParameteriv(
        GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &objectName);
    if (glGetError() != GL_NO_ERROR || objectType != GL_TEXTURE
        || objectName == 0)
        return false;

    GLint oldTexture = 0;
    GLint glWidth = 0;
    GLint glHeight = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTexture);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(objectName));
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &glWidth);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &glHeight);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(oldTexture));
    if (glGetError() != GL_NO_ERROR || glWidth <= 0 || glHeight <= 0)
        return false;

    texture = static_cast<GLuint>(objectName);
    width = static_cast<unsigned int>(glWidth);
    height = static_cast<unsigned int>(glHeight);
    return true;
}

bool CopyToOpenGlTexture(
    const GLuint texture, cudaArray_t source,
    const unsigned int width, const unsigned int height)
{
    cudaGraphicsResource_t resource = nullptr;
    if (source == nullptr)
        return false;
    if (cudaGraphicsGLRegisterImage(
            &resource, texture, GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        // An EGL/headless context cannot necessarily be associated with
        // CUDA. Consume only the interop error before using the CUDA renderer
        // again; the caller will use its bulk-copy fallback.
        cudaGetLastError();
        return false;
    }

    bool success = false;
    if (cudaGraphicsMapResources(1, &resource) == cudaSuccess) {
        cudaArray_t destination = nullptr;
        if (cudaGraphicsSubResourceGetMappedArray(
                &destination, resource, 0, 0) == cudaSuccess) {
            success = cudaMemcpy2DArrayToArray(
                destination, 0, 0, source, 0, 0,
                static_cast<size_t>(width) * sizeof(float) * 4, height,
                cudaMemcpyDeviceToDevice) == cudaSuccess;
        }
        cudaGraphicsUnmapResources(1, &resource);
    }
    if (cudaGraphicsUnregisterResource(resource) != cudaSuccess)
        cudaGetLastError();
    return success;
}

void UploadToOpenGlTexture(
    const GLuint texture, const Bitmap& bitmap,
    const unsigned int width, const unsigned int height)
{
    GLint oldTexture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTexture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(
        GL_TEXTURE_2D, 0, 0, 0, static_cast<GLsizei>(width),
        static_cast<GLsizei>(height), GL_RGBA, GL_FLOAT,
        bitmap.pixels().data());
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(oldTexture));
}

bool PresentBitmap(
    const Bitmap& bitmap, const bool directViewport,
    const GLuint glColorTexture, HdNoorRayRenderBuffer* colorBuffer,
    const unsigned int width, const unsigned int height)
{
    if (directViewport) {
        UploadToOpenGlTexture(glColorTexture, bitmap, width, height);
        return true;
    }
    if (colorBuffer != nullptr) {
        for (size_t pixel = 0; pixel < bitmap.pixels().size(); ++pixel)
            colorBuffer->WriteFloatPixel(
                pixel, &bitmap.pixels()[pixel].x, 4);
        return true;
    }
    return false;
}

bool PresentOutput(
    const Image& outputImage, const bool directViewport,
    const GLuint glColorTexture, HdNoorRayRenderBuffer* colorBuffer,
    const unsigned int width, const unsigned int height)
{
    if (outputImage.getCudaArray() == nullptr)
        return false;

    const bool sameSize = outputImage.getWidth() == width
        && outputImage.getHeight() == height;
    if (sameSize) {
        bool copied = directViewport && CopyToOpenGlTexture(
            glColorTexture, outputImage.getCudaArray(), width, height);
        if (!copied && colorBuffer != nullptr)
            copied = colorBuffer->CopyFromCudaArray(outputImage.getCudaArray());
        if (copied)
            return true;
    }

    const Bitmap bitmap = outputImage.toBitmap();
    if (!sameSize) {
        std::vector<glm::vec4> resized(static_cast<size_t>(width) * height);
        if (stbir_resize_float_linear(
                bitmap.rgba(), static_cast<int>(bitmap.width()),
                static_cast<int>(bitmap.height()), 0,
                reinterpret_cast<float*>(resized.data()),
                static_cast<int>(width), static_cast<int>(height), 0,
                STBIR_RGBA) == nullptr)
            return false;
        return PresentBitmap(
            Bitmap(width, height, std::move(resized)), directViewport,
            glColorTexture, colorBuffer, width, height);
    }
    return PresentBitmap(
        bitmap, directViewport, glColorTexture, colorBuffer, width, height);
}
}

HdNoorRayRenderPass::HdNoorRayRenderPass(
    HdRenderIndex* index,
    const HdRprimCollection& collection,
    HdNoorRayRenderParam& renderParam)
    : HdRenderPass(index, collection)
    , renderParam_(renderParam)
{
}

HdNoorRayRenderPass::~HdNoorRayRenderPass()
{
    _ReleaseViewportPresenter();
}

void HdNoorRayRenderPass::_ReleaseViewportPresenter()
{
    if (viewportCopyStream_ != nullptr)
        cudaStreamSynchronize(viewportCopyStream_);
    if (offlinePresentationPending_ && offlinePresentationReadyEvent_ != nullptr)
        cudaEventSynchronize(offlinePresentationReadyEvent_);
    if (viewportGraphicsResource_ != nullptr) {
        cudaGraphicsUnregisterResource(viewportGraphicsResource_);
        viewportGraphicsResource_ = nullptr;
    }
    if (viewportImage_ != nullptr) {
        cudaFreeArray(viewportImage_);
        viewportImage_ = nullptr;
    }
    if (viewportImageReadyEvent_ != nullptr) {
        cudaEventDestroy(viewportImageReadyEvent_);
        viewportImageReadyEvent_ = nullptr;
    }
    if (viewportCopyFinishedEvent_ != nullptr) {
        cudaEventDestroy(viewportCopyFinishedEvent_);
        viewportCopyFinishedEvent_ = nullptr;
    }
    if (viewportCopyStream_ != nullptr) {
        cudaStreamDestroy(viewportCopyStream_);
        viewportCopyStream_ = nullptr;
    }
    if (offlineStaging_ != nullptr) {
        cudaFreeHost(offlineStaging_);
        offlineStaging_ = nullptr;
    }
    if (offlinePresentationReadyEvent_ != nullptr) {
        cudaEventDestroy(offlinePresentationReadyEvent_);
        offlinePresentationReadyEvent_ = nullptr;
    }
    offlineStagingCapacity_ = 0;
    offlinePresentationPending_ = false;
    viewportImageWidth_ = 0;
    viewportImageHeight_ = 0;
    viewportImageValid_ = false;
    viewportImageEventRecorded_ = false;
    viewportCopyFinishedEventRecorded_ = false;
    viewportGraphicsTexture_ = 0;
    viewportGraphicsWidth_ = 0;
    viewportGraphicsHeight_ = 0;
}

bool HdNoorRayRenderPass::_EnsureViewportPresenter(
    const unsigned int texture, const unsigned int width, const unsigned int height)
{
    if (texture == 0 || width == 0 || height == 0)
        return false;

    if (viewportCopyStream_ == nullptr
        && cudaStreamCreateWithFlags(
               &viewportCopyStream_, cudaStreamNonBlocking) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (viewportImageReadyEvent_ == nullptr
        && cudaEventCreateWithFlags(
               &viewportImageReadyEvent_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (viewportCopyFinishedEvent_ == nullptr
        && cudaEventCreateWithFlags(
               &viewportCopyFinishedEvent_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const bool targetChanged = viewportGraphicsResource_ != nullptr
        && (viewportGraphicsTexture_ != texture
            || viewportGraphicsWidth_ != width
            || viewportGraphicsHeight_ != height);
    if (targetChanged) {
        if (cudaStreamSynchronize(viewportCopyStream_) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        cudaGraphicsUnregisterResource(viewportGraphicsResource_);
        viewportGraphicsResource_ = nullptr;
        viewportGraphicsTexture_ = 0;
    }
    if (viewportGraphicsResource_ == nullptr) {
        if (cudaGraphicsGLRegisterImage(
                &viewportGraphicsResource_, texture, GL_TEXTURE_2D,
                cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
            cudaGetLastError();
            viewportGraphicsResource_ = nullptr;
            return false;
        }
        viewportGraphicsTexture_ = texture;
        viewportGraphicsWidth_ = width;
        viewportGraphicsHeight_ = height;
    }

    const bool imageChanged = viewportImage_ == nullptr
        || viewportImageWidth_ != width || viewportImageHeight_ != height;
    if (imageChanged) {
        if (cudaStreamSynchronize(viewportCopyStream_) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (viewportImage_ != nullptr)
            cudaFreeArray(viewportImage_);
        const cudaChannelFormatDesc format = cudaCreateChannelDesc<float4>();
        if (cudaMallocArray(
                &viewportImage_, &format, width, height,
                cudaArraySurfaceLoadStore) != cudaSuccess) {
            cudaGetLastError();
            viewportImage_ = nullptr;
            viewportImageWidth_ = 0;
            viewportImageHeight_ = 0;
            viewportImageValid_ = false;
            return false;
        }
        viewportImageWidth_ = width;
        viewportImageHeight_ = height;
        viewportImageValid_ = false;
        viewportImageEventRecorded_ = false;
        viewportCopyFinishedEventRecorded_ = false;
    }
    return true;
}

bool HdNoorRayRenderPass::_QueueViewportImage(
    const cudaArray_t source, const unsigned int width,
    const unsigned int height, const cudaStream_t renderStream)
{
    if (source == nullptr || viewportImage_ == nullptr
        || viewportImageWidth_ != width || viewportImageHeight_ != height
        || viewportImageReadyEvent_ == nullptr)
        return false;

    // The OpenGL copy runs on a separate CUDA stream and reads viewportImage_;
    // do not overwrite that reusable staging array until the previous copy has
    // completed.
    if (viewportCopyFinishedEventRecorded_
        && cudaStreamWaitEvent(
               renderStream, viewportCopyFinishedEvent_, 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    cudaMemcpy3DParms copy{};
    copy.srcArray = source;
    copy.dstArray = viewportImage_;
    copy.extent = make_cudaExtent(width, height, 1);
    copy.kind = cudaMemcpyDeviceToDevice;
    if (cudaMemcpy3DAsync(&copy, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaEventRecord(viewportImageReadyEvent_, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    viewportImageValid_ = true;
    viewportImageEventRecorded_ = true;
    return true;
}

bool HdNoorRayRenderPass::_QueueViewportBuffer(
    const void* source, const unsigned int width,
    const unsigned int height, const cudaStream_t renderStream)
{
    if (source == nullptr || viewportImage_ == nullptr
        || viewportImageWidth_ != width || viewportImageHeight_ != height
        || viewportImageReadyEvent_ == nullptr)
        return false;

    if (viewportCopyFinishedEventRecorded_
        && cudaStreamWaitEvent(
               renderStream, viewportCopyFinishedEvent_, 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const size_t rowBytes = static_cast<size_t>(width) * sizeof(float) * 4;
    if (cudaMemcpy2DToArrayAsync(
            viewportImage_, 0, 0, source, rowBytes, rowBytes, height,
            cudaMemcpyDeviceToDevice, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaEventRecord(viewportImageReadyEvent_, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    viewportImageValid_ = true;
    viewportImageEventRecorded_ = true;
    return true;
}

bool HdNoorRayRenderPass::_PresentViewportImage(
    const unsigned int texture, const unsigned int width, const unsigned int height)
{
    if (!viewportImageValid_ || viewportImage_ == nullptr
        || viewportGraphicsResource_ == nullptr
        || viewportCopyStream_ == nullptr)
        return false;

    if (viewportImageEventRecorded_
        && cudaStreamWaitEvent(
               viewportCopyStream_, viewportImageReadyEvent_, 0) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaGraphicsMapResources(
            1, &viewportGraphicsResource_, viewportCopyStream_) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    cudaArray_t destination = nullptr;
    bool success = cudaGraphicsSubResourceGetMappedArray(
        &destination, viewportGraphicsResource_, 0, 0) == cudaSuccess;
    if (success) {
        cudaMemcpy3DParms copy{};
        copy.srcArray = viewportImage_;
        copy.dstArray = destination;
        copy.extent = make_cudaExtent(width, height, 1);
        copy.kind = cudaMemcpyDeviceToDevice;
        success = cudaMemcpy3DAsync(
            &copy, viewportCopyStream_) == cudaSuccess;
    }
    const cudaError_t unmapStatus = cudaGraphicsUnmapResources(
        1, &viewportGraphicsResource_, viewportCopyStream_);
    if (unmapStatus != cudaSuccess)
        success = false;
    if (success && cudaEventRecord(
            viewportCopyFinishedEvent_, viewportCopyStream_) != cudaSuccess) {
        cudaGetLastError();
        success = false;
    } else if (success)
        viewportCopyFinishedEventRecorded_ = true;
    if (!success)
        cudaGetLastError();
    return success;
}

bool HdNoorRayRenderPass::_EnsureOfflineStaging(const size_t bytes)
{
    if (bytes == 0)
        return false;
    if (offlinePresentationReadyEvent_ == nullptr
        && cudaEventCreateWithFlags(
               &offlinePresentationReadyEvent_, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (offlineStagingCapacity_ >= bytes)
        return true;

    if (offlinePresentationPending_)
        cudaEventSynchronize(offlinePresentationReadyEvent_);
    if (offlineStaging_ != nullptr)
        cudaFreeHost(offlineStaging_);
    offlineStaging_ = nullptr;
    offlineStagingCapacity_ = 0;
    if (cudaHostAlloc(&offlineStaging_, bytes, cudaHostAllocPortable) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    offlineStagingCapacity_ = bytes;
    return true;
}

bool HdNoorRayRenderPass::_QueueOfflinePresentation(
    const cudaArray_t source, const unsigned int width,
    const unsigned int height, const cudaStream_t renderStream)
{
    const size_t rowBytes = static_cast<size_t>(width) * sizeof(float) * 4;
    const size_t bytes = rowBytes * height;
    if (source == nullptr || !_EnsureOfflineStaging(bytes)
        || offlinePresentationReadyEvent_ == nullptr)
        return false;

    if (cudaMemcpy2DFromArrayAsync(
            offlineStaging_, rowBytes, source, 0, 0, rowBytes, height,
            cudaMemcpyDeviceToHost, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaEventRecord(offlinePresentationReadyEvent_, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    offlinePresentationPending_ = true;
    return true;
}

bool HdNoorRayRenderPass::_QueueOfflinePresentation(
    const void* source, const unsigned int width,
    const unsigned int height, const cudaStream_t renderStream)
{
    const size_t rowBytes = static_cast<size_t>(width) * sizeof(float) * 4;
    const size_t bytes = rowBytes * height;
    if (source == nullptr || !_EnsureOfflineStaging(bytes)
        || offlinePresentationReadyEvent_ == nullptr)
        return false;

    if (cudaMemcpy2DAsync(
            offlineStaging_, rowBytes, source, rowBytes, rowBytes, height,
            cudaMemcpyDeviceToHost, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (cudaEventRecord(offlinePresentationReadyEvent_, renderStream) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    offlinePresentationPending_ = true;
    return true;
}

bool HdNoorRayRenderPass::_CommitOfflinePresentation(
    HdNoorRayRenderBuffer& colorBuffer, const size_t bytes)
{
    if (!offlinePresentationPending_ || offlinePresentationReadyEvent_ == nullptr
        || offlineStaging_ == nullptr || bytes > offlineStagingCapacity_)
        return false;
    const cudaError_t status = cudaEventQuery(offlinePresentationReadyEvent_);
    if (status == cudaErrorNotReady)
        return false;
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    void* destination = colorBuffer.Map();
    if (destination == nullptr)
        return false;
    std::memcpy(destination, offlineStaging_, bytes);
    colorBuffer.Unmap();
    offlinePresentationPending_ = false;
    return true;
}

void HdNoorRayRenderPass::SetBuffersConverged(
    const HdRenderPassAovBindingVector& bindings, const bool converged)
{
    for (const HdRenderPassAovBinding& binding : bindings)
        if (HdNoorRayRenderBuffer* buffer = GetBuffer(binding))
            buffer->SetConverged(converged);
}

bool HdNoorRayRenderPass::IsConverged() const
{
    return converged_;
}

void HdNoorRayRenderPass::_MarkCollectionDirty()
{
    collectionDirty_ = true;
}

void HdNoorRayRenderPass::_Execute(
    const HdRenderPassStateSharedPtr& renderPassState,
    const TfTokenVector&)
{
    // Hydra calls this from the host's render loop, and an exception thrown
    // through that boundary terminates the host (Blender) outright. Anything
    // the renderer can fail on — a GPU allocation, an OptiX launch, a scene
    // that cannot be sized — is contained here instead.
    //
    // Deliberately does NOT report convergence on failure. That used to be
    // the behavior here (stop asking for a frame this pass cannot produce,
    // on the theory that a later scene/camera change would reset converged_
    // and give it another attempt), but Blender's own Hydra viewport engine
    // stops scheduling continuous redraws once the pass reports converged --
    // so a transient failure (e.g. hitting this mid-resync, while Blender's
    // full scene re-export is mid-flight tearing down and rebuilding Sprims,
    // see HdNoorRayRenderParam::ReleaseMaterial's comment) reported as
    // "converged" could leave the viewport stuck on whatever was last drawn
    // (often nothing), with no guaranteed later call giving it the "another
    // attempt" the old comment assumed would come. Leaving
    // converged_ alone means _Render's own reset logic (sceneVersion/camera/
    // size checks, already re-evaluated on every call) decides when to
    // retry, and the render loop keeps calling us until it actually
    // succeeds instead of giving up after the first failure.
    try {
        _Render(renderPassState);
    } catch (const std::exception& error) {
        // TF_RUNTIME_ERROR alone is not reliably visible in Blender's own
        // console (it goes through USD's diagnostic delegate, which Blender
        // does not always surface) -- write directly to stderr too so a
        // render failure is never silent.
        fprintf(stderr, "[hdNoorRay] could not render the frame: %s\n",
            error.what());
        TF_RUNTIME_ERROR(
            "hdNoorRay could not render the frame: %s", error.what());
        SetBuffersConverged(renderPassState->GetAovBindings(), converged_);
    } catch (...) {
        fprintf(stderr, "[hdNoorRay] could not render the frame: unknown exception\n");
        TF_RUNTIME_ERROR("hdNoorRay could not render the frame");
        SetBuffersConverged(renderPassState->GetAovBindings(), converged_);
    }
}

void HdNoorRayRenderPass::_Render(
    const HdRenderPassStateSharedPtr& renderPassState)
{
    const HdRenderPassAovBindingVector& bindings =
        renderPassState->GetAovBindings();
    HdNoorRayRenderBuffer* colorBuffer = nullptr;
    for (const HdRenderPassAovBinding& binding : bindings) {
        if (binding.aovName == HdAovTokens->color)
            colorBuffer = GetBuffer(binding);
    }
    GLuint glColorTexture = 0;
    unsigned int width = colorBuffer != nullptr ? colorBuffer->GetWidth() : 0;
    unsigned int height = colorBuffer != nullptr ? colorBuffer->GetHeight() : 0;
    const bool directViewport = colorBuffer == nullptr
        && GetOpenGlColorTarget(glColorTexture, width, height);
    if ((!directViewport && colorBuffer == nullptr) || width == 0 || height == 0) {
        fprintf(stderr,
            "[hdNoorRay] skipping render: no usable render target yet "
            "(colorBuffer=%p directViewport=%d width=%u height=%u)\n",
            static_cast<void*>(colorBuffer), directViewport, width, height);
        return;
    }

    // Material compilation runs on TBB workers. OptiX objects and scene
    // bindings are committed here in one render-thread batch.
    const bool compiledMaterialsChanged =
        renderParam_.ProcessMaterialCompilations();
    std::unique_lock lock(renderParam_.mutex);
    Scene& scene = renderParam_.session.scene;
    if (renderParam_.session.raytracer == nullptr) {
        // Deliberately does NOT report convergence here (that used to be
        // the behavior, on the theory that a GPU-less session can never
        // render so the host might as well stop asking) -- see
        // HdNoorRayRenderPass::_Execute's comment for why forcing converged_
        // true on a "cannot render right now" condition can get the viewport
        // stuck: Blender's Hydra viewport stops scheduling this pass
        // continuously once it reports converged, so if this is ever hit
        // transiently (e.g. a session recreation still constructing its
        // Raytracer when this runs) rather than a genuinely GPU-less host,
        // there is no guaranteed later call to notice the raytracer exists
        // and recover. Logging loudly instead of silently giving up is the
        // point -- this used to return with no output at all.
        fprintf(stderr,
            "[hdNoorRay] skipping render: no Raytracer on this session yet "
            "(GPU init still running, or failed -- see earlier log output)\n");
        SetBuffersConverged(bindings, converged_);
        return;
    }
    Raytracer& raytracer = *renderParam_.session.raytracer;

    HdRenderDelegate* delegate = GetRenderIndex()->GetRenderDelegate();
    const int targetSamples = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("samples"), 64));

    // Hydra keeps calling Execute while a render buffer is unconverged.
    // Material compilation can take seconds on a large import, and rendering
    // during that interval only produces fallback frames which are discarded
    // when the completed material batch changes the scene version.  An
    // offline render has no interactive preview to preserve, so do no GPU
    // work until its materials are ready.  The viewport may show its initial
    // progressive preview, but once it reaches the requested sample count it
    // also stops launching redundant frames while compilation finishes.
    const bool materialsPending =
        renderParam_.HasPendingMaterialCompilations();
    if (materialsPending
        && (!directViewport
            || accumulatedSamples_ >= static_cast<unsigned int>(targetSamples))) {
        if (directViewport)
            PresentOutput(
                raytracer.getOutputColor(), true, glColorTexture, nullptr,
                width, height);
        converged_ = false;
        SetBuffersConverged(bindings, false);
        return;
    }

    const unsigned int sceneVersion =
        GetRenderIndex()->GetChangeTracker().GetSceneStateVersion();
    const unsigned int renderSettingsVersion =
        delegate->GetRenderSettingsVersion();
    const GfMatrix4d newProjection = renderPassState->GetProjectionMatrix();
    const HdCamera* hydraCamera = renderPassState->GetCamera();
    const GfMatrix4d newCameraTransform = hydraCamera != nullptr
        ? hydraCamera->GetTransform()
        : renderPassState->GetWorldToViewMatrix().GetInverse();
    const bool cameraChanged = cameraTransform_ != newCameraTransform
        || projectionMatrix_ != newProjection;

    // A resized target has to reach the sensor even when the camera itself did
    // not move, or the renderer keeps producing frames at the previous
    // resolution and the copy back into the render buffer reads the wrong size.
    const bool sizeChanged = width != targetWidth_ || height != targetHeight_;
    const bool renderSettingsChanged =
        observedRenderSettingsVersion_ != renderSettingsVersion;
    const bool sceneVersionChanged = observedSceneVersion_ != sceneVersion;
    const bool reset = collectionDirty_ || cameraChanged || sizeChanged
        || renderSettingsChanged || sceneVersionChanged
        || compiledMaterialsChanged;
    if (reset) {
        accumulatedSamples_ = 0;
        converged_ = false;
        renderParam_.ResetClock();
    }

    bool completedViewportFramePresented = false;
    // A heavy full-resolution OptiX launch must not stall Blender's viewport
    // event loop.  Keep at most one launch in flight: while it is running,
    // return to Blender so navigation input and Hydra sync can continue.  On
    // the first later Execute after completion, present that completed frame
    // before submitting the newest scene/camera state.  This deliberately
    // does not change render resolution, samples, or shading quality.
    if (directViewport && framePending_) {
        if (raytracer.isRenderInFlight()) {
            // Blender clears its direct viewport target between Execute calls.
            // Replay the retained completed frame while CUDA produces the
            // next one, rather than leaving the viewport blank.
            _PresentViewportImage(glColorTexture, width, height);
            converged_ = false;
            SetBuffersConverged(bindings, false);
            return;
        }
        if (!_PresentViewportImage(glColorTexture, width, height)) {
            // CUDA/GL interop can be unavailable on unusual GL contexts.
            // This fallback is safe now because the render has completed.
            PresentOutput(
                raytracer.getOutputColor(), true, glColorTexture, nullptr,
                width, height);
        }
        renderParam_.AccumulateGpuTimeMs(raytracer.getGpuTimeMs());
        framePending_ = false;
        completedViewportFramePresented = true;
    }

    // A host may queue more Execute calls before it observes the converged
    // render-buffer flag. Do not turn those already-queued calls into extra
    // samples after the requested total is complete.
    if (!reset
        && accumulatedSamples_ >= static_cast<unsigned int>(targetSamples)) {
        if (directViewport && !completedViewportFramePresented)
            PresentOutput(
                raytracer.getOutputColor(), true, glColorTexture, nullptr,
                width, height);
        renderParam_.SetProgress(1.0);
        converged_ = true;
        SetBuffersConverged(bindings, true);
        return;
    }

    CameraInstance* cameraInstance = scene.getRenderCamera();
    if (cameraInstance == nullptr)
        return;
    Camera* camera = cameraInstance->getCamera();

    // Apply NoorRay camera projection (overrides USD orthographic hint).
    bool projectionSwitched = false;
    {
        const int projectionSetting = delegate->GetRenderSetting<int>(
            TfToken("cameraProjection"), -1);
        const CameraProjectionType projection = projectionSetting >= 0
            ? static_cast<CameraProjectionType>(projectionSetting)
            : hydraCamera != nullptr
                  && hydraCamera->GetProjection() == HdCamera::Orthographic
              ? CameraProjectionType::Orthographic
              : CameraProjectionType::Perspective;
        if (cameraInstance->getProjectionType() != projection) {
            cameraInstance->switchTo(projection);
            camera = cameraInstance->getCamera();
            projectionSwitched = true;
        }
    }
    const bool opticalCamera = camera->Is<RealisticCamera>()
        || camera->Is<HybridPsfCamera>();

    if (cameraChanged || reset || projectionSwitched) {
        const int sensorTypeSetting = delegate->GetRenderSetting<int>(
            TfToken("cameraSensorType"), -1);
        if (sensorTypeSetting >= 0) {
            const SensorType requestedType = static_cast<SensorType>(sensorTypeSetting);
            if (camera->getSensor().getType() != requestedType) {
                switch (requestedType) {
                case SensorType::Rectangular:
                    camera->setSensor(std::make_unique<RectangularSensor>(camera->getSensor()));
                    break;
                case SensorType::ScatterPsf:
                    camera->setSensor(std::make_unique<ScatterPsfSensor>(camera->getSensor()));
                    break;
                case SensorType::GatherPsf:
                    camera->setSensor(std::make_unique<GatherPsfSensor>(camera->getSensor()));
                    break;
                }
            }
        }

        Sensor& sensor = camera->getSensor();
        const std::string sensorPath = delegate->GetRenderSetting<std::string>(
            TfToken("cameraSensorPath"), {});
        if (sensorPath != sensor.getImageSensorPath()) {
            sensor.setImageSensorPath(sensorPath);
            if (!sensorPath.empty())
                sensor.loadImageSensorDimensions();
        }
        const std::string psfPath = delegate->GetRenderSetting<std::string>(
            TfToken("cameraPsfPath"), {});
        if (sensor.getType() != SensorType::Rectangular
            && psfPath != sensor.getPsfGridPath())
            sensor.loadPsfGrid(psfPath);

        const float sensorWidthMm = delegate->GetRenderSetting<float>(
            TfToken("cameraSensorWidthMm"), -1.0f);
        const float sensorHeightMm = delegate->GetRenderSetting<float>(
            TfToken("cameraSensorHeightMm"), -1.0f);
        const bool hasRenderSettingFilm =
            sensorWidthMm > 0.0f && sensorHeightMm > 0.0f;
        if (hasRenderSettingFilm) {
            sensor.setDimensionsMm(sensorWidthMm, sensorHeightMm);
        } else if (hydraCamera != nullptr) {
            // Compatibility fallback for non-Blender Hydra clients.
            sensor.setDimensionsMm(
                hydraCamera->GetHorizontalAperture(),
                hydraCamera->GetVerticalAperture());
        }
        // Optical cameras own their sampling grid: loading the image sensor
        // above establishes its authoritative resolution. Ordinary cameras
        // follow Blender's current render/viewport target as before.
        if (!opticalCamera)
            sensor.setResolution(width, height);

        // Blender sends its native lens in millimeters as a render setting.
        // Hydra camera geometry may use a host-scaled film unit (Blender's is
        // commonly one tenth of a millimeter). Never combine the Hydra film
        // dimensions with the millimeter focal length: that changes the
        // focal/film ratio and produces an approximately 10x zoom.
        const float focalLengthMm = delegate->GetRenderSetting<float>(
            TfToken("cameraFocalLengthMm"), -1.0f);
        const bool hasHydraCameraGeometry = hydraCamera != nullptr
            && hydraCamera->GetHorizontalAperture() > 0.0
            && hydraCamera->GetVerticalAperture() > 0.0
            && hydraCamera->GetFocalLength() > 0.0;
        if (!opticalCamera) {
            if (focalLengthMm > 0.0f)
                camera->setFocalLengthMm(focalLengthMm);
            else if (hasHydraCameraGeometry)
                camera->setFocalLengthMm(hydraCamera->GetFocalLength());
        }

        const float exposure = delegate->GetRenderSetting<float>(
            TfToken("cameraExposure"), std::numeric_limits<float>::quiet_NaN());
        if (std::isfinite(exposure))
            camera->setExposure(exposure);

        const float apertureDiameterMm = delegate->GetRenderSetting<float>(
            TfToken("cameraApertureDiameter"), -1.0f);
        if (apertureDiameterMm >= 0.0f) {
            if (auto* tl = camera->CastOrNullptr<ThinLensCamera>())
                tl->apertureDiameterMm = apertureDiameterMm;
            else if (auto* fi = camera->CastOrNullptr<FisheyeCamera>())
                fi->apertureDiameterMm = apertureDiameterMm;
            else if (auto* re = camera->CastOrNullptr<RealisticCamera>())
                re->setApertureDiameterMm(apertureDiameterMm);
            else if (auto* hp = camera->CastOrNullptr<HybridPsfCamera>())
                hp->setApertureDiameterMm(apertureDiameterMm);
        }
        const float bokehBias = delegate->GetRenderSetting<float>(
            TfToken("cameraBokehBias"), -1.0f);
        if (bokehBias >= 0.0f) {
            if (auto* tl = camera->CastOrNullptr<ThinLensCamera>())
                tl->bokehBias = bokehBias;
            else if (auto* fi = camera->CastOrNullptr<FisheyeCamera>())
                fi->bokehBias = bokehBias;
        }
        const std::string lensPath = delegate->GetRenderSetting<std::string>(
            TfToken("cameraLensPath"), {});
        const std::string glassCatalogs =
            delegate->GetRenderSetting<std::string>(
                TfToken("cameraGlassCatalogs"), {});
        const std::string rayLutPath = delegate->GetRenderSetting<std::string>(
            TfToken("cameraRayLutPath"), {});
        const int rayLutStepSize = delegate->GetRenderSetting<int>(
            TfToken("cameraRayLutStepSize"), -1);
        const int apertureSamplesPerDimension = delegate->GetRenderSetting<int>(
            TfToken("cameraApertureSamplesPerDimension"), -1);
        if (auto* re = camera->CastOrNullptr<RealisticCamera>()) {
            if (lensPath != re->getLensPath()
                || glassCatalogs != re->getGlassCatalogPaths())
                re->load(lensPath, glassCatalogs);
        } else if (auto* hp = camera->CastOrNullptr<HybridPsfCamera>()) {
            const int previousRayLutStepSize = hp->rayLutStepSize;
            const int previousApertureSamplesPerDimension = hp->samplesPerDimension;
            const int requestedRayLutStepSize = rayLutStepSize > 0
                ? rayLutStepSize : previousRayLutStepSize;
            const int requestedApertureSamplesPerDimension =
                apertureSamplesPerDimension > 0
                ? apertureSamplesPerDimension : previousApertureSamplesPerDimension;
            if (requestedRayLutStepSize != previousRayLutStepSize
                || requestedApertureSamplesPerDimension
                    != previousApertureSamplesPerDimension) {
                nr::synchronizeBeforeManagedMutation("Hybrid PSF LUT settings");
                hp->rayLutStepSize = requestedRayLutStepSize;
                hp->samplesPerDimension = requestedApertureSamplesPerDimension;
            }
            const bool lutSettingsChanged =
                previousRayLutStepSize != hp->rayLutStepSize
                || previousApertureSamplesPerDimension != hp->samplesPerDimension;
            if (lensPath != hp->getLensPath()
                || glassCatalogs != hp->getGlassCatalogPaths()
                || rayLutPath != hp->getRayLutPath())
                hp->load(lensPath, glassCatalogs, rayLutPath);
            else if (lutSettingsChanged && !hp->getLensPath().empty())
                hp->loadLensSensorAndPsf();
        }

        // Apply this after loading an image sensor: optical loaders restore
        // the physical dimensions, while the fitted film is render-target
        // dependent and must be derived from the current Hydra size.
        const int sensorFitSetting = delegate->GetRenderSetting<int>(
            TfToken("cameraSensorFit"), 1);
        const SensorFit sensorFit = static_cast<SensorFit>(std::clamp(
            sensorFitSetting, 0, 2));
        sensor.setFilmFit(sensorFit, width, height);

        // Blender's Hydra projection already includes its exact sensor-fit,
        // pixel-aspect, and target-size calculation. Use it to remove small
        // focal/zoom discrepancies caused by reimplementing that calculation
        // from rounded camera properties. Stretch intentionally keeps the
        // legacy raw-sensor behavior and therefore skips this correction.
        const bool perspectiveFilm = !opticalCamera
            && cameraInstance->getProjectionType()
                != CameraProjectionType::Orthographic;
        const double projectionScaleX = std::abs(newProjection[0][0]);
        const double projectionScaleY = std::abs(newProjection[1][1]);
        const float projectionFocalLengthMm = camera->getFocalLengthMm();
        if (perspectiveFilm && sensorFit != SensorFit::Stretch
            && std::isfinite(projectionScaleX)
            && std::isfinite(projectionScaleY)
            && projectionScaleX > 0.0 && projectionScaleY > 0.0
            && std::isfinite(projectionFocalLengthMm)
            && projectionFocalLengthMm > 0.0f) {
            sensor.setFilmDimensionsMm(
                static_cast<float>(2.0 * projectionFocalLengthMm / projectionScaleX),
                static_cast<float>(2.0 * projectionFocalLengthMm / projectionScaleY));
        }

        const float focusDistanceCm = delegate->GetRenderSetting<float>(
            TfToken("cameraFocusDistanceCm"), -1.0f);
        if (focusDistanceCm > 0.0f)
            camera->setFocusDistanceCm(focusDistanceCm);

        cameraInstance->setWorldTransformFromMatrix(ToGlm(newCameraTransform));
        sensor.setOrigin(SensorOrigin::LowerLeft);
    }

    RenderSettings& rs = scene.getRenderSettings();
    // Submit one sample per pass for both viewport and offline Blender
    // renders. Hydra/F12 must receive the intermediate framebuffer so the
    // user can see convergence instead of waiting for the complete sample
    // budget before the first image is displayed.
    const unsigned int samplesThisPass = 1u;
    rs.samples = static_cast<int>(samplesThisPass);
    rs.maxSamples = targetSamples;
    rs.maxBounces = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("maxBounces"), 8));
    rs.indirectLightClamp = std::max(
        0.0f, delegate->GetRenderSetting<float>(
            TfToken("indirectLightClamp"), 10.0f));
    rs.optixDenoiserEnabled = delegate->GetRenderSetting<int>(
        TfToken("optixDenoiserEnabled"), 0) != 0;
    rs.optixDenoiserMinSamples = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("optixDenoiserMinSamples"), 1));
    // The delegate hands Blender scene-linear radiance and lets its colour
    // management own the display transform, so the renderer never tonemaps.
    rs.tonemappingEnabled = false;
    rs.cameraExposure = delegate->GetRenderSetting<float>(
        TfToken("cameraExposure"), 0.0f);
    rs.transparentBackground = delegate->GetRenderSetting<int>(
        TfToken("transparentBackground"), 0) != 0;
    rs.gaussianCutoffSigma = std::max(
        0.1f, delegate->GetRenderSetting<float>(TfToken("gaussianCutoffSigma"), 3.0f));
    rs.gaussianProxyType = static_cast<GaussianProxyType>(
        delegate->GetRenderSetting<int>(TfToken("gaussianProxyType"), 3));
    rs.gaussianShadingMode = static_cast<GaussianShadingMode>(
        delegate->GetRenderSetting<int>(TfToken("gaussianShadingMode"), 1));
    rs.gaussianRenderSphericalHarmonics = static_cast<SphericalHarmonicsOrder>(
        std::clamp(delegate->GetRenderSetting<int>(TfToken("gaussianRenderSphericalHarmonics"), 3), 0, 3));
    rs.bufferVisualization = static_cast<BufferVisualization>(
        std::clamp(delegate->GetRenderSetting<int>(TfToken("bufferVisualization"), 0), 0, 6));
    rs.gaussianProxyOverdrawVisualization = delegate->GetRenderSetting<int>(
        TfToken("gaussianProxyOverdrawVisualization"), 0) != 0;
    rs.gaussianProxyOverdrawMax = std::clamp(delegate->GetRenderSetting<int>(
        TfToken("gaussianProxyOverdrawMax"), 1024), 1, 1024 * 1024);
    // AOVs are expensive full-resolution side buffers. The Hydra integration
    // has no consumer for them except the OptiX denoiser's albedo and normal
    // guides, so keep both gates off by default and enable them together only
    // when denoising is actually requested.
    const bool needsDenoiserAovs = runsOptixDenoiser(rs);
    rs.aovEnabled = needsDenoiserAovs;
    raytracer.setAovEnabled(needsDenoiserAovs);
    raytracer.renderFrame(accumulatedSamples_, accumulatedSamples_);
    if (directViewport) {
        // Keep a dedicated CUDA staging image for presentation. Blender
        // clears the OpenGL viewport target before each Execute, while this
        // retained image lets us replay the last finished full-resolution
        // frame without ever reading the output being written by OptiX.
        const bool queuedForPresentation = _EnsureViewportPresenter(
                glColorTexture, width, height)
            && _QueueViewportImage(
                raytracer.getOutputColor().getCudaArray(), width, height,
                raytracer.getCudaStream());
        // The completion event is polled at the top of the next Execute.
        // Waiting here made a costly frame block camera navigation entirely.
        // If interop setup failed, the completion path uses its safe fallback.
        (void)queuedForPresentation;
        framePending_ = true;
    } else {
        // Offline/F12 rendering owns a CPU-side render buffer, so it must
        // wait for and copy this sample before returning to its caller.
        raytracer.waitForRender();
        renderParam_.AccumulateGpuTimeMs(raytracer.getGpuTimeMs());
        PresentOutput(
            raytracer.getOutputColor(), false, glColorTexture,
            colorBuffer, width, height);
    }
    accumulatedSamples_ += samplesThisPass;
    converged_ = !directViewport
        && accumulatedSamples_ >= static_cast<unsigned int>(targetSamples)
        && !renderParam_.HasPendingMaterialCompilations();
    // Defer non-zero progress until after the first frame so Blender's
    // render engine can initialize its wall-clock time baseline.
    if (accumulatedSamples_ > 1)
        renderParam_.SetProgress(
            static_cast<double>(accumulatedSamples_) / targetSamples);
    SetBuffersConverged(bindings, converged_);

    observedSceneVersion_ = sceneVersion;
    observedRenderSettingsVersion_ = renderSettingsVersion;
    cameraTransform_ = newCameraTransform;
    projectionMatrix_ = newProjection;
    targetWidth_ = width;
    targetHeight_ = height;
    collectionDirty_ = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
