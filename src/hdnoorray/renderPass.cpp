#include "renderPass.h"

#include "renderBuffer.h"
#include "renderParam.h"

#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/camera.h>
#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
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
    const HdRenderPassAovBindingVector& bindings =
        renderPassState->GetAovBindings();
    HdNoorRayRenderBuffer* colorBuffer = nullptr;
    bool depthRequested = false;
    for (const HdRenderPassAovBinding& binding : bindings) {
        if (binding.aovName == HdAovTokens->color)
            colorBuffer = GetBuffer(binding);
        else if (binding.aovName == HdAovTokens->depth)
            depthRequested = GetBuffer(binding) != nullptr;
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
    Raytracer& raytracer = *renderParam_.session.raytracer;

    const uint64_t sceneVersion = renderParam_.GetSceneVersion();
    const GfMatrix4d newProjection = renderPassState->GetProjectionMatrix();
    const HdCamera* hydraCamera = renderPassState->GetCamera();
    const GfMatrix4d newCameraTransform = hydraCamera != nullptr
        ? hydraCamera->GetTransform()
        : renderPassState->GetWorldToViewMatrix().GetInverse();
    const bool cameraChanged = cameraTransform_ != newCameraTransform
        || projectionMatrix_ != newProjection;
    const bool reset = collectionDirty_ || cameraChanged
        || observedSceneVersion_ != sceneVersion;
    if (reset) {
        accumulatedSamples_ = 0;
        converged_ = false;
    }

    CameraInstance* cameraInstance = scene.getRenderCamera();
    if (cameraInstance == nullptr)
        return;
    Camera* camera = cameraInstance->getCamera();
    const CameraProjectionType projection = hydraCamera != nullptr
        && hydraCamera->GetProjection() == HdCamera::Orthographic
        ? CameraProjectionType::Orthographic
        : CameraProjectionType::Perspective;
    if (cameraInstance->getProjectionType() != projection) {
        cameraInstance->switchTo(projection);
        camera = cameraInstance->getCamera();
    }

    if (cameraChanged || reset) {
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
        cameraInstance->setWorldTransformFromMatrix(ToGlm(newCameraTransform));
        camera->getSensor().setOrigin(SensorOrigin::LowerLeft);
    }

    HdRenderDelegate* delegate = GetRenderIndex()->GetRenderDelegate();
    const int targetSamples = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("samples"), 64));
    RenderSettings& settings = scene.getRenderSettings();
    settings.samples = 1;
    settings.maxSamples = targetSamples;
    settings.maxBounces = std::max(
        1, delegate->GetRenderSetting<int>(TfToken("maxBounces"), 8));
    settings.aovEnabled = depthRequested && !directViewport;
    raytracer.setAovEnabled(depthRequested);

    raytracer.renderFrame(accumulatedSamples_, accumulatedSamples_);
    raytracer.waitForRender();

    const Image& colorImage = raytracer.getOutputColor();
    bool copied = directViewport && CopyToOpenGlTexture(
        glColorTexture, colorImage.getCudaArray(), width, height);
    if (!copied && colorBuffer != nullptr)
        copied = colorBuffer->CopyFromCudaArray(colorImage.getCudaArray());
    if (!copied) {
        const Bitmap color = colorImage.toBitmap();
        if (directViewport) {
            UploadToOpenGlTexture(glColorTexture, color, width, height);
            copied = true;
        } else if (colorBuffer != nullptr) {
            for (size_t pixel = 0; pixel < color.pixels().size(); ++pixel)
                colorBuffer->WriteFloatPixel(
                    pixel, &color.pixels()[pixel].x, 4);
        }
    }

    if (depthRequested) {
        const Bitmap position = raytracer.getOutputPosition().toBitmap();
        const GfMatrix4d view = renderPassState->GetWorldToViewMatrix();
        for (const HdRenderPassAovBinding& binding : bindings) {
            if (binding.aovName != HdAovTokens->depth)
                continue;
            HdNoorRayRenderBuffer* depthBuffer = GetBuffer(binding);
            if (depthBuffer == nullptr)
                continue;
            for (size_t pixel = 0; pixel < position.pixels().size(); ++pixel) {
                float depth = 1.0f;
                const glm::vec4& world = position.pixels()[pixel];
                if (world.w > 0.0f) {
                    GfVec3d point(world.x, world.y, world.z);
                    point = view.Transform(point);
                    point = newProjection.Transform(point);
                    depth = std::clamp(
                        static_cast<float>((point[2] + 1.0) * 0.5), 0.0f, 1.0f);
                }
                depthBuffer->WriteFloatPixel(pixel, &depth, 1);
            }
        }
    }

    ++accumulatedSamples_;
    converged_ = accumulatedSamples_ >= static_cast<unsigned int>(targetSamples);
    for (const HdRenderPassAovBinding& binding : bindings)
        if (HdNoorRayRenderBuffer* buffer = GetBuffer(binding))
            buffer->SetConverged(converged_);

    observedSceneVersion_ = sceneVersion;
    cameraTransform_ = newCameraTransform;
    projectionMatrix_ = newProjection;
    collectionDirty_ = false;
}

PXR_NAMESPACE_CLOSE_SCOPE
