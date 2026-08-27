#include "renderPass.h"

#include "renderBuffer.h"
#include "renderParam.h"

#include "Backend/Vulkan/Raytracer/RaytracerRenderer.h"
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/FisheyeCamera.h"
#include "Rendering/Camera/RealisticCamera.h"
#include "Rendering/Camera/ThinLensCamera.h"
#include <gpu/interop.hpp>

#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include <unistd.h>

PXR_NAMESPACE_OPEN_SCOPE

namespace {
glm::mat4 ToGlm(const GfMatrix4d& value)
{
    glm::mat4 result(1.0f);
    for (int row = 0; row != 4; ++row)
        for (int column = 0; column != 4; ++column)
            result[column][row] = static_cast<float>(value[column][row]);
    return result;
}

HdNoorRayRenderBuffer* GetBuffer(const HdRenderPassAovBinding& binding)
{
    return dynamic_cast<HdNoorRayRenderBuffer*>(binding.renderBuffer);
}

bool GetOpenGlColorTarget(GLuint& texture, unsigned int& width, unsigned int& height)
{
    GLint type = GL_NONE, name = 0;
    glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_TYPE, &type);
    glGetFramebufferAttachmentParameteriv(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
    if (glGetError() != GL_NO_ERROR || type != GL_TEXTURE || name == 0)
        return false;
    GLint oldTexture = 0, glWidth = 0, glHeight = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTexture);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(name));
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &glWidth);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &glHeight);
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(oldTexture));
    if (glGetError() != GL_NO_ERROR || glWidth <= 0 || glHeight <= 0)
        return false;
    texture = static_cast<GLuint>(name);
    width = static_cast<unsigned int>(glWidth);
    height = static_cast<unsigned int>(glHeight);
    return true;
}

void UploadBgra(const GLuint texture, const unsigned int width, const unsigned int height,
    const std::vector<std::byte>& pixels)
{
    GLint oldTexture = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTexture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_BGRA,
        GL_UNSIGNED_BYTE, pixels.data());
    glBindTexture(GL_TEXTURE_2D, static_cast<GLuint>(oldTexture));
}
} // namespace

HdNoorRayRenderPass::HdNoorRayRenderPass(HdRenderIndex* index,
    const HdRprimCollection& collection, HdNoorRayRenderParam& renderParam)
    : HdRenderPass(index, collection), renderParam_(renderParam) {}

HdNoorRayRenderPass::~HdNoorRayRenderPass() { _ReleaseInteropImage(); }

bool HdNoorRayRenderPass::IsConverged() const { return converged_; }
void HdNoorRayRenderPass::_MarkCollectionDirty() { collectionDirty_ = true; }

void HdNoorRayRenderPass::_ReleaseInteropImage()
{
    if (interopReadFramebuffer_)
        glDeleteFramebuffers(1, &interopReadFramebuffer_);
    if (interopTexture_)
        glDeleteTextures(1, &interopTexture_);
    if (interopMemory_)
        glDeleteMemoryObjectsEXT(1, &interopMemory_);
    interopMemory_ = interopTexture_ = interopReadFramebuffer_ = 0;
    interopWidth_ = interopHeight_ = 0;
}

bool HdNoorRayRenderPass::_EnsureInteropImage(const unsigned int width,
    const unsigned int height)
{
    if (interopTexture_ && interopWidth_ == width && interopHeight_ == height)
        return true;
    _ReleaseInteropImage();
    const char* extensions = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
    if (!extensions || !std::strstr(extensions, "GL_EXT_memory_object_fd")
        || !std::strstr(extensions, "GL_EXT_semaphore_fd"))
        return false;
    try {
        const auto exported = gpu::interop::export_image_memory(
            *renderParam_.session.device, renderParam_.session.raytracer->colorHandle());
        glCreateMemoryObjectsEXT(1, &interopMemory_);
        glImportMemoryFdEXT(interopMemory_, exported.allocation_size,
            GL_HANDLE_TYPE_OPAQUE_FD_EXT, exported.fd);
        glGenTextures(1, &interopTexture_);
        glBindTexture(GL_TEXTURE_2D, interopTexture_);
        glTexStorageMem2DEXT(GL_TEXTURE_2D, 1, GL_RGBA32F, width, height,
            interopMemory_, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glGenFramebuffers(1, &interopReadFramebuffer_);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, interopReadFramebuffer_);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D, interopTexture_, 0);
        const bool complete = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER)
            == GL_FRAMEBUFFER_COMPLETE;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        if (!complete || glGetError() != GL_NO_ERROR) {
            _ReleaseInteropImage();
            return false;
        }
        interopWidth_ = width;
        interopHeight_ = height;
        return true;
    } catch (...) {
        _ReleaseInteropImage();
        return false;
    }
}

bool HdNoorRayRenderPass::_PresentInterop(const unsigned int targetTexture,
    const unsigned int width, const unsigned int height, const int semaphoreFd)
{
    if (!interopTexture_ || semaphoreFd < 0)
        return false;
    GLuint semaphore = 0;
    glGenSemaphoresEXT(1, &semaphore);
    glImportSemaphoreFdEXT(semaphore, GL_HANDLE_TYPE_OPAQUE_FD_EXT, semaphoreFd);
    const GLenum layout = GL_LAYOUT_GENERAL_EXT;
    glWaitSemaphoreEXT(semaphore, 0, nullptr, 1, &interopTexture_, &layout);
    GLint previousRead = 0, previousDraw = 0;
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &previousRead);
    glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &previousDraw);
    GLuint draw = 0;
    glGenFramebuffers(1, &draw);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, interopReadFramebuffer_);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, draw);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D, targetTexture, 0);
    glBlitFramebuffer(0, 0, interopWidth_, interopHeight_, 0, 0, width, height,
        GL_COLOR_BUFFER_BIT, GL_NEAREST);
    // OpenGL owns the imported image until the blit has consumed it.  A
    // future frame will write the same Vulkan allocation again; without a
    // reverse external semaphore, finish this small presentation operation
    // before returning ownership to Vulkan.  This preserves correctness on
    // every GL context while avoiding a CPU pixel readback.
    glFinish();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, static_cast<GLuint>(previousRead));
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, static_cast<GLuint>(previousDraw));
    glDeleteFramebuffers(1, &draw);
    glDeleteSemaphoresEXT(1, &semaphore);
    return glGetError() == GL_NO_ERROR;
}

bool HdNoorRayRenderPass::_PresentLastFrame(const unsigned int targetTexture,
    const unsigned int width, const unsigned int height)
{
    auto& session = renderParam_.session;
    if (!session.raytracer)
        return false;
    if (_EnsureInteropImage(width, height)) {
        try {
            const auto semaphore = gpu::interop::signal_external(*session.device);
            if (_PresentInterop(targetTexture, width, height, semaphore.fd))
                return true;
        } catch (...) {
        }
    }
    session.raytracer->device().synchronize();
    UploadBgra(targetTexture, width, height, session.raytracer->readColor());
    return true;
}

void HdNoorRayRenderPass::SetBuffersConverged(
    const HdRenderPassAovBindingVector& bindings, const bool converged)
{
    for (const auto& binding : bindings)
        if (auto* buffer = GetBuffer(binding))
            buffer->SetConverged(converged);
}

void HdNoorRayRenderPass::_Execute(const HdRenderPassStateSharedPtr& state,
    const TfTokenVector&)
{
    try {
        _Render(state);
    } catch (const std::exception& error) {
        fprintf(stderr, "[hdNoorRay] Vulkan render failed: %s\n", error.what());
        TF_RUNTIME_ERROR("hdNoorRay Vulkan render failed: %s", error.what());
        SetBuffersConverged(state->GetAovBindings(), false);
    }
}

void HdNoorRayRenderPass::_Render(const HdRenderPassStateSharedPtr& state)
{
    static const bool debug = getenv("HDNOORRAY_DEBUG") != nullptr;
    const auto& bindings = state->GetAovBindings();
    HdNoorRayRenderBuffer* colorBuffer = nullptr;
    for (const auto& binding : bindings)
        if (binding.aovName == HdAovTokens->color)
            colorBuffer = GetBuffer(binding);
    const auto debugLog = [&](const char* what) {
        if (debug)
            fprintf(stderr, "[hdNoorRay] frame: %s (buffer=%p samples=%u)\n",
                what, static_cast<void*>(colorBuffer), accumulatedSamples_);
    };

    GLuint glTexture = 0;
    unsigned int width = colorBuffer ? colorBuffer->GetWidth() : 0;
    unsigned int height = colorBuffer ? colorBuffer->GetHeight() : 0;
    bool viewport = !colorBuffer && GetOpenGlColorTarget(glTexture, width, height);
    if (!colorBuffer && !viewport) {
        // Blender alternates between framebuffer bindings across redraws, and
        // on roughly every other Execute the colour attachment query comes
        // back empty. Doing nothing on those frames lets Blender's plain
        // viewport show through (flicker). Reuse the last texture the query
        // returned while it is still alive; it is the same target Blender
        // clears and expects us to repaint.
        if (cachedViewportTexture_ && glIsTexture(cachedViewportTexture_)) {
            glTexture = cachedViewportTexture_;
            width = cachedViewportWidth_;
            height = cachedViewportHeight_;
            viewport = true;
            debugLog("target query failed, using cached viewport texture");
        }
    }
    if (viewport && glTexture) {
        cachedViewportTexture_ = glTexture;
        cachedViewportWidth_ = width;
        cachedViewportHeight_ = height;
    }
    if (width == 0 || height == 0 || (!colorBuffer && !viewport)) {
        debugLog("skip: no usable render target");
        return;
    }

    // Material publication takes the render-param mutex itself. Complete that
    // batch before holding the scene lock for the Vulkan dispatch.
    const bool compiledMaterialsChanged = renderParam_.ProcessMaterialCompilations();
    std::unique_lock lock(renderParam_.mutex);
    auto& session = renderParam_.session;
    if (!session.raytracer) {
        session.initializeHeadlessRenderer(width, height, viewport);
        session.rebuildNativeScene();
    } else if (session.raytracer->width() != width || session.raytracer->height() != height)
        session.raytracer->resize(width, height);

    HdRenderDelegate* delegate = GetRenderIndex()->GetRenderDelegate();
    // Material compilation takes seconds on a large import, and rendering
    // during that interval only produces fallback frames that get discarded
    // when the completed batch changes the scene state. Present the retained
    // last-good frame instead -- Blender clears its viewport target between
    // Execute calls, so going quiet here shows through as flicker.
    const unsigned int targetSamples = std::max(1,
        delegate->GetRenderSetting<int>(TfToken("samples"), 64));
    if (renderParam_.HasPendingMaterialCompilations()
        && (!viewport || accumulatedSamples_ >= targetSamples)) {
        debugLog("materials pending, presenting retained frame");
        if (viewport)
            _PresentLastFrame(glTexture, width, height);
        converged_ = false;
        renderParam_.SetProgress(accumulatedSamples_
            ? std::min(1.0, static_cast<double>(accumulatedSamples_) / targetSamples)
            : 0.0);
        SetBuffersConverged(bindings, false);
        return;
    }

    const auto sceneVersion = GetRenderIndex()->GetChangeTracker().GetSceneStateVersion();
    const auto settingsVersion = delegate->GetRenderSettingsVersion();
    const GfMatrix4d projection = state->GetProjectionMatrix();
    const HdCamera* hydraCamera = state->GetCamera();
    const GfMatrix4d transform = hydraCamera ? hydraCamera->GetTransform()
        : state->GetWorldToViewMatrix().GetInverse();
    const bool reset = collectionDirty_ || sceneVersion != observedSceneVersion_
        || settingsVersion != observedRenderSettingsVersion_ || width != targetWidth_
        || height != targetHeight_ || projection != projectionMatrix_
        || transform != cameraTransform_ || compiledMaterialsChanged;
    if (reset) {
        debugLog(collectionDirty_ ? "reset: collection dirty"
            : compiledMaterialsChanged ? "reset: materials changed"
            : sceneVersion != observedSceneVersion_ ? "reset: scene version"
            : settingsVersion != observedRenderSettingsVersion_ ? "reset: render settings"
            : width != targetWidth_ || height != targetHeight_ ? "reset: size"
            : projection != projectionMatrix_ || transform != cameraTransform_
                ? "reset: camera" : "reset");
        accumulatedSamples_ = 0; converged_ = false; renderParam_.ResetClock();
    }

    if (CameraInstance* instance = session.scene.getRenderCamera()) {
        const int requestedProjection = delegate->GetRenderSetting<int>(
            TfToken("cameraProjection"), -1);
        // Hybrid PSF was a legacy camera. The Vulkan backend uses the
        // same physical-lens snapshot through RealisticCamera until a native
        // PSF evaluator is available, preserving the lens/DOF path instead
        // of silently falling back to pinhole projection.
        const CameraProjectionType projectionType = requestedProjection == 5
            ? CameraProjectionType::Realistic
            : requestedProjection >= 0
            && requestedProjection <= static_cast<int>(CameraProjectionType::Realistic)
            ? static_cast<CameraProjectionType>(requestedProjection)
            : hydraCamera && hydraCamera->GetProjection() == HdCamera::Orthographic
                ? CameraProjectionType::Orthographic
                : CameraProjectionType::Perspective;
        instance->switchTo(projectionType);
        Camera* camera = instance->getCamera();
        camera->getSensor().setResolution(width, height);
        if (hydraCamera && hydraCamera->GetFocalLength() > 0.0)
            camera->setFocalLengthMm(static_cast<float>(hydraCamera->GetFocalLength()));
        camera->setExposure(delegate->GetRenderSetting<float>(TfToken("cameraExposure"), 0.0f));
        const float focusDistance = delegate->GetRenderSetting<float>(
            TfToken("cameraFocusDistanceCm"), -1.0f);
        if (focusDistance > 0.0f)
            camera->setFocusDistanceCm(focusDistance);
        const float aperture = std::max(0.0f, delegate->GetRenderSetting<float>(
            TfToken("cameraApertureDiameter"), 0.0f));
        if (auto* thinLens = camera->CastOrNullptr<ThinLensCamera>())
            thinLens->apertureDiameterMm = aperture;
        else if (auto* fisheye = camera->CastOrNullptr<FisheyeCamera>())
            fisheye->apertureDiameterMm = aperture;
        else if (auto* realistic = camera->CastOrNullptr<RealisticCamera>()) {
            realistic->setApertureDiameterMm(aperture);
            const std::string lens = delegate->GetRenderSetting<std::string>(
                TfToken("cameraLensPath"), {});
            const std::string catalogs = delegate->GetRenderSetting<std::string>(
                TfToken("cameraGlassCatalogs"), {});
            if (lens != realistic->getLensPath()
                || catalogs != realistic->getGlassCatalogPaths())
                realistic->load(lens, catalogs);
        }
        instance->setWorldTransformFromMatrix(ToGlm(transform));
        session.updateNativeCamera();
    }

    RenderSettings& settings = session.scene.getRenderSettings();
    settings.samples = 1;
    settings.maxSamples = static_cast<int>(targetSamples);
    settings.maxBounces = std::max(1, delegate->GetRenderSetting<int>(TfToken("maxBounces"), 8));
    settings.indirectLightClamp = std::max(0.0f, delegate->GetRenderSetting<float>(TfToken("indirectLightClamp"), 10.0f));
    settings.aovEnabled = delegate->GetRenderSetting<int>(TfToken("aovEnabled"), 1) != 0;
    settings.transparentBackground = delegate->GetRenderSetting<int>(TfToken("transparentBackground"), 0) != 0;
    settings.gaussianCutoffSigma = std::max(0.1f, delegate->GetRenderSetting<float>(TfToken("gaussianCutoffSigma"), 3.0f));
    settings.gaussianProxyType = static_cast<GaussianProxyType>(std::clamp(
        delegate->GetRenderSetting<int>(TfToken("gaussianProxyType"), 3), 0, 3));
    settings.gaussianShadingMode = static_cast<GaussianShadingMode>(std::clamp(
        delegate->GetRenderSetting<int>(TfToken("gaussianShadingMode"), 1), 0, 1));
    settings.gaussianRenderSphericalHarmonics = static_cast<SphericalHarmonicsOrder>(std::clamp(
        delegate->GetRenderSetting<int>(TfToken("gaussianRenderSphericalHarmonics"), 3), 0, 3));
    settings.gaussianProxyOverdrawVisualization = delegate->GetRenderSetting<int>(
        TfToken("gaussianProxyOverdrawVisualization"), 0) != 0;
    settings.gaussianProxyOverdrawMax = std::max(1, delegate->GetRenderSetting<int>(
        TfToken("gaussianProxyOverdrawMax"), 1024));
    settings.bufferVisualization = static_cast<BufferVisualization>(std::clamp(
        delegate->GetRenderSetting<int>(TfToken("bufferVisualization"), 0), 0, 5));

    if (reset)
        session.raytracer->uploadScene(session.scene);

    if (accumulatedSamples_ < targetSamples) {
        session.pollNativeScene();
        session.raytracer->render(accumulatedSamples_, accumulatedSamples_);
        ++accumulatedSamples_;
    }
    bool interopPresented = false;
    if (viewport && accumulatedSamples_ > 0) {
        interopPresented = _PresentLastFrame(glTexture, width, height);
        if (debug)
            fprintf(stderr, "[hdNoorRay] frame: present %s (interop=%d)\n",
                interopPresented ? "ok" : "fallback", interopPresented ? 1 : 0);
    }
    if (!interopPresented)
        session.raytracer->device().synchronize();
    renderParam_.AccumulateGpuTimeMs(static_cast<float>(session.raytracer->lastDispatchMilliseconds()));
    if (colorBuffer) {
        const auto beauty = session.raytracer->readBeauty();
        colorBuffer->CopyFromHost(beauty.data(), beauty.size() * sizeof(beauty.front()));
    }
    // Blender stops scheduling viewport redraws once the pass reports
    // converged -- but it also clears its direct viewport target between
    // Execute calls, and unrelated UI redraws clear it again without a
    // matching Execute. Reporting convergence in the viewport therefore lets
    // the image get wiped with no one left to repaint it (flicker). The old
    // OptiX pass never converged in direct-viewport mode for exactly this
    // reason; keep that behavior.
    converged_ = !viewport && accumulatedSamples_ >= targetSamples
        && !renderParam_.HasPendingMaterialCompilations();
    // Defer non-zero progress until after the first frame so Blender's
    // render engine can initialize its wall-clock time baseline.
    if (accumulatedSamples_ > 1)
        renderParam_.SetProgress(static_cast<double>(accumulatedSamples_) / targetSamples);
    SetBuffersConverged(bindings, converged_);
    observedSceneVersion_ = sceneVersion; observedRenderSettingsVersion_ = settingsVersion;
    targetWidth_ = width; targetHeight_ = height; cameraTransform_ = transform;
    projectionMatrix_ = projection; collectionDirty_ = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
