#include "CameraInstance.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "Scene/SceneObjectVisitor.h"
#include "Rendering/Camera/RealisticCamera.h"

void CameraInstance::accept(SceneObjectVisitor& visitor)
{
    visitor.visit(static_cast<SceneObject&>(*this));
    visitor.visit(*this);
}
#include "Backend/Host/MutationBarrier.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"
#include "Scene/Scene.h"

// ── unified-memory allocation ─────────────────────────────────────────────────

void CameraInstance::allocateCamera(CameraProjectionType type)
{
    nr::synchronizeBeforeManagedMutation("Camera projection replacement");

    Camera state = camera->cloneBaseState();
    std::unique_ptr<Sensor> transferredSensor = camera->releaseSensor();
    std::string sharedLensPath;
    std::string sharedGlassCatalogPaths;
    if (const auto* realistic = camera->CastOrNullptr<RealisticCamera>()) {
        sharedLensPath = realistic->getLensPath();
        sharedGlassCatalogPaths = realistic->getGlassCatalogPaths();
    }
    std::unique_ptr<Camera> replacement;
    switch (type) {
    case CameraProjectionType::ThinLens:
        replacement = std::make_unique<ThinLensCamera>(std::move(transferredSensor)); break;
    case CameraProjectionType::Orthographic:
        replacement = std::make_unique<OrthographicCamera>(std::move(transferredSensor)); break;
    case CameraProjectionType::Fisheye:
        replacement = std::make_unique<FisheyeCamera>(std::move(transferredSensor)); break;
    case CameraProjectionType::Realistic:
        replacement = std::make_unique<RealisticCamera>(std::move(transferredSensor)); break;
    case CameraProjectionType::Perspective:
        replacement = std::make_unique<PerspectiveCamera>(std::move(transferredSensor)); break;
    }
    camera = std::move(replacement);
    static_cast<Camera&>(*camera) = state;

    if (auto* realistic = camera->CastOrNullptr<RealisticCamera>())
        realistic->setOpticsPaths(sharedLensPath, sharedGlassCatalogPaths);

}

// ── constructor / destructor ──────────────────────────────────────────────────

CameraInstance::CameraInstance(
    std::unique_ptr<Camera> ownedCamera, const std::string& name, Transform transform)
    : SceneObject(name, transform), camera(std::move(ownedCamera))
{
    if (!camera)
        throw std::invalid_argument("CameraInstance requires a Camera");
    if (!*camera)
        throw std::invalid_argument("CameraInstance requires a tagged concrete Camera type");
    rebuildCamera();
}

CameraInstance::CameraInstance(const CameraInstance& other)
    : SceneObject(other),
      arcballPivot(other.arcballPivot),
      arcballMode(other.arcballMode)
{
    nr::synchronizeBeforeManagedMutation("Camera clone");
    if (const auto* source = other.camera->CastOrNullptr<PerspectiveCamera>())
        camera = std::make_unique<PerspectiveCamera>(*source);
    else if (const auto* source = other.camera->CastOrNullptr<ThinLensCamera>())
        camera = std::make_unique<ThinLensCamera>(*source);
    else if (const auto* source = other.camera->CastOrNullptr<OrthographicCamera>())
        camera = std::make_unique<OrthographicCamera>(*source);
    else if (const auto* source = other.camera->CastOrNullptr<FisheyeCamera>())
        camera = std::make_unique<FisheyeCamera>(*source);
    else if (const auto* source = other.camera->CastOrNullptr<RealisticCamera>())
        camera = std::make_unique<RealisticCamera>(*source);
    else
        camera = std::make_unique<Camera>(*other.camera);
    rebuildCamera();
}

CameraInstance::~CameraInstance()
{
    // Render kernels dereference the camera and its sensor from managed memory.
    // Retire any frame that still references them before unique_ptr frees either.
    nr::synchronizeBeforeManagedMutation("Camera destruction");
}

// ── core ──────────────────────────────────────────────────────────────────────

void CameraInstance::markDirty()
{
    if (!scene) return;
    scene->setDirtyFlag(CameraState);
    scene->setDirtyFlag(Accumulation);
}

void CameraInstance::rebuildCamera()
{
    const quat  rot    = getRotation();
    const vec3  dir    = normalize(rot * LocalForward);
    const vec3  up     = normalize(rot * LocalUp);
    const vec3  right  = normalize(cross(dir, up));

    const mat4 cameraToWorld = mat4(
        vec4(right,         0.f),
        vec4(up,            0.f),
        vec4(-dir,          0.f),
        vec4(getPosition(), 1.f));

    camera->setCameraToWorld(cameraToWorld);
}

void CameraInstance::onTransformUpdated()
{
    SceneObject::onTransformUpdated();
    rebuildCamera();
    markDirty();
}

void CameraInstance::switchTo(CameraProjectionType type)
{
    if (getProjectionType() == type)
        return;
    allocateCamera(type);
    rebuildCamera();
    // Replacing the tagged camera changes the device-visible camera pointer
    // and invalidates both beauty accumulation and camera-dependent AOVs.
    // The replacement itself is synchronized in allocateCamera().
    markDirty();
}

std::unique_ptr<SceneObject> CameraInstance::clone() const
{
    return std::make_unique<CameraInstance>(*this);
}

// ── queries ──────────────────────────────────────────────────────────────────

CameraProjectionType CameraInstance::getProjectionType() const
{
    if (camera->Is<ThinLensCamera>())    return CameraProjectionType::ThinLens;
    if (camera->Is<OrthographicCamera>()) return CameraProjectionType::Orthographic;
    if (camera->Is<FisheyeCamera>())     return CameraProjectionType::Fisheye;
    if (camera->Is<RealisticCamera>())   return CameraProjectionType::Realistic;
    return CameraProjectionType::Perspective;
}

const char* CameraInstance::getProjectionName() const
{
    if (camera->Is<ThinLensCamera>())    return "Thin Lens";
    if (camera->Is<OrthographicCamera>()) return "Orthographic";
    if (camera->Is<FisheyeCamera>())     return "Fisheye";
    if (camera->Is<RealisticCamera>())   return "Realistic";
    return "Perspective";
}

mat4 CameraInstance::getViewMatrix() const
{
    const vec3 dir = normalize(getRotation() * LocalForward);
    const vec3 up  = normalize(getRotation() * LocalUp);
    return lookAt(getPosition(), getPosition() + dir, up);
}

mat4 CameraInstance::getProjectionMatrix() const
{
    const Sensor& sensor = camera->getSensor();
    const float aspect = sensor.filmWidth() / sensor.filmHeight();
    const float focalLengthMm = camera->getFocalLengthMm();
    const float fovY = 2.f * std::atan(sensor.filmHeight() / (2.f * focalLengthMm));
    return perspective(fovY, aspect, 0.01f, 10000.f);
}

// ── arcball ──────────────────────────────────────────────────────────────────

void CameraInstance::setArcballPivot(const vec3& pivot)
{
    arcballPivot = pivot;
}

// ── update (input) ───────────────────────────────────────────────────────────

void CameraInstance::update(const float dx, const float dy, const InputState& input)
{
    const vec3 oldPosition = getPosition();
    const quat oldRotation = getRotation();

    if (arcballMode) {
        vec3 position    = getPosition();
        quat orientation = getRotation();
        float moveSpeed  = input.deltaTime * 5.f;
        if (input.accelerate) moveSpeed *= 10.f;

        const vec3 dirToCamera = normalize(position - arcballPivot);
        if (input.forward) position -= dirToCamera * moveSpeed;
        if (input.backward) position += dirToCamera * moveSpeed;

        constexpr float sensitivity = 0.004f;
        const quat yawQuat   = angleAxis(dx * sensitivity, WorldUp);
        const quat pitchQuat = angleAxis(dy * sensitivity, orientation * LocalRight);
        vec3 offset = position - arcballPivot;
        offset = yawQuat * pitchQuat * offset;
        setPosition(arcballPivot + offset);
        setRotation(normalize(yawQuat * pitchQuat * orientation));
    } else {
        constexpr float sensitivity = 0.1f;
        const float yaw   = radians(-dx * sensitivity);
        const float pitch = radians(-dy * sensitivity);
        quat rot = getRotation();
        const quat yawQuat   = angleAxis(yaw, WorldUp);
        const quat pitchQuat = angleAxis(pitch, rot * LocalRight);
        const quat newRot    = normalize(yawQuat * pitchQuat * rot);
        setRotation(newRot);

        float speed = input.deltaTime * 5.f;
        if (input.accelerate) speed *= 10.f;

        vec3 position   = getPosition();
        const vec3 fwd  = normalize(newRot * LocalForward);
        const vec3 up   = normalize(newRot * LocalUp);
        const vec3 rgt  = normalize(cross(fwd, up));

        if (input.forward) position += fwd * speed;
        if (input.backward) position -= fwd * speed;
        if (input.left) position -= rgt * speed;
        if (input.right) position += rgt * speed;
        if (input.up) position += up  * speed;
        if (input.down) position -= up  * speed;
        setPosition(position);
    }

    if (oldPosition != getPosition() || oldRotation != getRotation())
        markDirty();
}
