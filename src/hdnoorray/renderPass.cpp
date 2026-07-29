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
#include <cstdio>
#include <exception>
#include <mutex>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include "Camera/Camera.h"
#include "Camera/CameraInstance.h"
#include "IO/Bitmap.h"
#include "Raytracing/Runtime/Raytracer.h"
#include "Scene/Scene.h"
#include "Vulkan/Image.h"

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

bool PresentOutput(
    const Image& outputImage, const bool directViewport,
    const GLuint glColorTexture, HdNoorRayRenderBuffer* colorBuffer,
    const unsigned int width, const unsigned int height)
{
    if (outputImage.getCudaArray() == nullptr
        || outputImage.getWidth() != width
        || outputImage.getHeight() != height)
        return false;

    bool copied = directViewport && CopyToOpenGlTexture(
        glColorTexture, outputImage.getCudaArray(), width, height);
    if (!copied && colorBuffer != nullptr)
        copied = colorBuffer->CopyFromCudaArray(outputImage.getCudaArray());
    if (copied)
        return true;

    const Bitmap bitmap = outputImage.toBitmap();
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
}

HdNoorRayRenderPass::HdNoorRayRenderPass(
    HdRenderIndex* index,
    const HdRprimCollection& collection,
    HdNoorRayRenderParam& renderParam)
    : HdRenderPass(index, collection)
    , renderParam_(renderParam)
{
}

HdNoorRayRenderPass::~HdNoorRayRenderPass() = default;

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

    // OSL compilation runs on TBB workers. OptiX objects and scene bindings
    // are committed here, in one render-thread batch, without ever blocking
    // Hydra Sync or Blender's UI.
    const bool compiledMaterialsChanged =
        renderParam_.ProcessMaterialCompilations();
    std::scoped_lock lock(renderParam_.mutex);
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
        if (directViewport) {
            // Blender clears its GPU AOV before every viewport draw. Keep the
            // last progressive image visible while the replacement materials
            // finish compiling, even though no new sample is launched.
            PresentOutput(
                raytracer.getOutputColor(), true, glColorTexture, nullptr,
                width, height);
        }
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
    // A host may queue more Execute calls before it observes the converged
    // render-buffer flag. Do not turn those already-queued calls into extra
    // samples after the requested total is complete.
    if (!reset
        && accumulatedSamples_ >= static_cast<unsigned int>(targetSamples)) {
        if (directViewport) {
            // Blender's GPU render-task delegate has just cleared this
            // texture. Re-present the retained final image without exceeding
            // the configured sample limit.
            PresentOutput(
                raytracer.getOutputColor(), true, glColorTexture, nullptr,
                width, height);
        }
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

    if (cameraChanged || reset || projectionSwitched) {
        camera->getSensor().setResolution(
            width, height);
        if (hydraCamera != nullptr) {
            camera->getSensor().setDimensionsMm(
                hydraCamera->GetHorizontalAperture(),
                hydraCamera->GetVerticalAperture());
            camera->setFocalLengthMm(hydraCamera->GetFocalLength());
            camera->setFocusDistanceCm(
                hydraCamera->GetFocusDistance() * 100.0f);
        }

        const float apertureDiameterMm = delegate->GetRenderSetting<float>(
            TfToken("cameraApertureDiameter"), -1.0f);
        if (apertureDiameterMm >= 0.0f) {
            if (auto* tl = camera->CastOrNullptr<ThinLensCamera>())
                tl->apertureDiameterMm = apertureDiameterMm;
            else if (auto* fi = camera->CastOrNullptr<FisheyeCamera>())
                fi->apertureDiameterMm = apertureDiameterMm;
            else if (auto* re = camera->CastOrNullptr<RealisticCamera>())
                re->apertureDiameterMm = apertureDiameterMm;
            else if (auto* hp = camera->CastOrNullptr<HybridPsfCamera>())
                hp->apertureDiameterMm = apertureDiameterMm;
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
        if (!lensPath.empty()) {
            if (auto* re = camera->CastOrNullptr<RealisticCamera>()) {
                if (lensPath != re->getLensPath()
                    || glassCatalogs != re->getGlassCatalogPaths())
                    re->load(lensPath, glassCatalogs);
            } else if (auto* hp = camera->CastOrNullptr<HybridPsfCamera>()) {
                if (lensPath != hp->getLensPath()
                    || glassCatalogs != hp->getGlassCatalogPaths())
                    hp->load(lensPath, glassCatalogs, rayLutPath);
            }
        }

        cameraInstance->setWorldTransformFromMatrix(ToGlm(newCameraTransform));
        camera->getSensor().setOrigin(SensorOrigin::LowerLeft);
    }

    RenderSettings& rs = scene.getRenderSettings();
    // Offline Blender renders do not benefit from copying every intermediate
    // sample through the Hydra render buffer.  Submit all remaining samples
    // in this pass and copy the converged image once.  Keep viewport updates
    // at one sample per pass for interactivity.
    const unsigned int remainingSamples =
        accumulatedSamples_ < static_cast<unsigned int>(targetSamples)
        ? static_cast<unsigned int>(targetSamples) - accumulatedSamples_
        : 1u;
    const unsigned int samplesThisPass =
        directViewport ? 1u : remainingSamples;
    rs.samples = static_cast<int>(samplesThisPass);
    rs.maxSamples = targetSamples;
    rs.maxBounces = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("maxBounces"), 8));
    rs.noiseLimitEnabled = delegate->GetRenderSetting<int>(
        TfToken("noiseLimitEnabled"), 0) != 0;
    rs.noiseLevel = delegate->GetRenderSetting<float>(
        TfToken("noiseLevel"), 0.0001f);
    rs.optixDenoiserEnabled = delegate->GetRenderSetting<int>(
        TfToken("optixDenoiserEnabled"), 0) != 0;
    rs.optixDenoiserMinSamples = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("optixDenoiserMinSamples"), 1));
    rs.russianRouletteStartBounce = std::clamp(
        delegate->GetRenderSetting<int>(TfToken("russianRouletteStartBounce"), 3),
        0, 65);
    // The delegate hands Blender scene-linear radiance and lets its colour
    // management own the display transform, so the renderer never tonemaps.
    rs.tonemappingEnabled = false;
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
    rs.gaussianProxyOverdrawVisualization = delegate->GetRenderSetting<int>(
        TfToken("gaussianProxyOverdrawVisualization"), 0) != 0;
    // hdNoorRay only ever hands Hydra a colour buffer, so the denoiser's
    // albedo and normal guides are the sole use it has for AOVs. Asking for
    // them here is what allocates their images; with the denoiser off nothing
    // does, and the AOV raygen never launches.
    raytracer.setAovEnabled(rs.optixDenoiserEnabled);
    raytracer.renderFrame(accumulatedSamples_, accumulatedSamples_);
    raytracer.waitForRender();
    renderParam_.AccumulateGpuTimeMs(raytracer.getGpuTimeMs());

    PresentOutput(
        raytracer.getOutputColor(), directViewport, glColorTexture,
        colorBuffer, width, height);

    accumulatedSamples_ += samplesThisPass;
    converged_ = accumulatedSamples_ >= static_cast<unsigned int>(targetSamples)
        && !renderParam_.HasPendingMaterialCompilations();
    // Defer non-zero progress until after the first frame so Blender's
    // viewport engine can initialize its wall-clock time baseline (it only
    // does so when percentDone == 0, checked after engine_->Execute()).
    if (accumulatedSamples_ > 1) {
        renderParam_.SetProgress(
            static_cast<double>(accumulatedSamples_) / targetSamples);
    }
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
