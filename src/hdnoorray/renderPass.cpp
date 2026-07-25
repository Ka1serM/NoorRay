#include "renderPass.h"

#include "renderBuffer.h"
#include "renderParam.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
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
    try {
        _Render(renderPassState);
    } catch (const std::exception& error) {
        TF_RUNTIME_ERROR(
            "hdNoorRay could not render the frame: %s", error.what());
        // Report convergence so the host stops asking for a frame that this
        // pass cannot produce. A later scene or camera change clears it and
        // the renderer gets another attempt.
        converged_ = true;
        SetBuffersConverged(renderPassState->GetAovBindings(), true);
    } catch (...) {
        TF_RUNTIME_ERROR("hdNoorRay could not render the frame");
        converged_ = true;
        SetBuffersConverged(renderPassState->GetAovBindings(), true);
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
    if ((!directViewport && colorBuffer == nullptr) || width == 0 || height == 0)
        return;

    std::scoped_lock lock(renderParam_.mutex);
    Scene& scene = renderParam_.session.scene;
    if (renderParam_.session.raytracer == nullptr) {
        // No usable GPU context. Report convergence so the host stops asking
        // for frames instead of spinning on a pass that can never finish.
        converged_ = true;
        SetBuffersConverged(bindings, true);
        return;
    }
    Raytracer& raytracer = *renderParam_.session.raytracer;

    HdRenderDelegate* delegate = GetRenderIndex()->GetRenderDelegate();
    const uint64_t sceneVersion = renderParam_.GetSceneVersion();
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
    const bool reset = collectionDirty_ || cameraChanged || sizeChanged
        || renderParam_.ConsumeRenderSettingsChanged()
        || observedSceneVersion_ != sceneVersion;
    if (reset) {
        accumulatedSamples_ = 0;
        converged_ = false;
        renderParam_.ResetClock();
    }

    CameraInstance* cameraInstance = scene.getRenderCamera();
    if (cameraInstance == nullptr)
        return;
    Camera* camera = cameraInstance->getCamera();

    // Apply NoorRay camera projection (overrides USD orthographic hint).
    bool projectionSwitched = false;
    {
        const CameraSettings& cs = renderParam_.cameraSettings;
        const CameraProjectionType projection = cs.projectionType >= 0
            ? static_cast<CameraProjectionType>(cs.projectionType)
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

        const CameraSettings& cs = renderParam_.cameraSettings;
        if (cs.apertureDiameterMm >= 0.0f) {
            if (auto* tl = camera->CastOrNullptr<ThinLensCamera>())
                tl->apertureDiameterMm = cs.apertureDiameterMm;
            else if (auto* fi = camera->CastOrNullptr<FisheyeCamera>())
                fi->apertureDiameterMm = cs.apertureDiameterMm;
            else if (auto* re = camera->CastOrNullptr<RealisticCamera>())
                re->apertureDiameterMm = cs.apertureDiameterMm;
            else if (auto* hp = camera->CastOrNullptr<HybridPsfCamera>())
                hp->apertureDiameterMm = cs.apertureDiameterMm;
        }
        if (cs.bokehBias >= 0.0f) {
            if (auto* tl = camera->CastOrNullptr<ThinLensCamera>())
                tl->bokehBias = cs.bokehBias;
            else if (auto* fi = camera->CastOrNullptr<FisheyeCamera>())
                fi->bokehBias = cs.bokehBias;
        }
        if (!cs.lensPath.empty()) {
            if (auto* re = camera->CastOrNullptr<RealisticCamera>()) {
                if (cs.lensPath != re->getLensPath()
                    || cs.glassCatalogs != re->getGlassCatalogPaths())
                    re->load(cs.lensPath, cs.glassCatalogs);
            } else if (auto* hp = camera->CastOrNullptr<HybridPsfCamera>()) {
                if (cs.lensPath != hp->getLensPath()
                    || cs.glassCatalogs != hp->getGlassCatalogPaths())
                    hp->load(cs.lensPath, cs.glassCatalogs, cs.rayLutPath);
            }
        }

        cameraInstance->setWorldTransformFromMatrix(ToGlm(newCameraTransform));
        camera->getSensor().setOrigin(SensorOrigin::LowerLeft);
    }

    const int targetSamples = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("samples"), 64));
    RenderSettings& rs = scene.getRenderSettings();
    rs.samples = 1;
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

    const Image* outputImage = &raytracer.getOutputColor();
    bool copied = directViewport && CopyToOpenGlTexture(
        glColorTexture, outputImage->getCudaArray(), width, height);
    if (!copied && colorBuffer != nullptr)
        copied = colorBuffer->CopyFromCudaArray(outputImage->getCudaArray());
    if (!copied) {
        const Bitmap bitmap = outputImage->toBitmap();
        if (directViewport) {
            UploadToOpenGlTexture(glColorTexture, bitmap, width, height);
            copied = true;
        } else if (colorBuffer != nullptr) {
            for (size_t pixel = 0; pixel < bitmap.pixels().size(); ++pixel)
                colorBuffer->WriteFloatPixel(
                    pixel, &bitmap.pixels()[pixel].x, 4);
        }
    }

    ++accumulatedSamples_;
    converged_ = accumulatedSamples_ >= static_cast<unsigned int>(targetSamples);
    // Defer non-zero progress until after the first frame so Blender's
    // viewport engine can initialize its wall-clock time baseline (it only
    // does so when percentDone == 0, checked after engine_->Execute()).
    if (accumulatedSamples_ > 1) {
        renderParam_.SetProgress(
            static_cast<double>(accumulatedSamples_) / targetSamples);
    }
    SetBuffersConverged(bindings, converged_);

    observedSceneVersion_ = sceneVersion;
    cameraTransform_ = newCameraTransform;
    projectionMatrix_ = newProjection;
    targetWidth_ = width;
    targetHeight_ = height;
    collectionDirty_ = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
