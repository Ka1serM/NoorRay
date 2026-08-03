#include "Rendering/Camera/Camera.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <type_traits>

#include "Backend/CUDA/ManagedMemory.h"

Camera::~Camera() = default;

Camera::Camera(std::unique_ptr<Sensor> ownedSensor)
    : sensor(ownedSensor.release())
{
    if (!sensor)
        throw std::invalid_argument("Camera requires a Sensor");
    if (!*sensor)
        throw std::invalid_argument("Camera requires a tagged concrete Sensor type");
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        sensor->getPsfGridPath().c_str());
}

Camera::Camera(const Camera& other)
    : TaggedCamera(other), cameraToWorld(other.cameraToWorld)
    , focalLengthMm(other.focalLengthMm)
    , focusDistanceCm(other.focusDistanceCm), exposure(other.exposure)
{
    const Sensor& source = other.getSensor();
    source.DispatchCPU([this, &source](const auto* concrete) {
        using SensorType = std::remove_cvref_t<decltype(*concrete)>;
        sensor.reset(new SensorType(source));
    });
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        other.retainedPsfGridPath);
}

Camera& Camera::operator=(const Camera& other)
{
    if (this == &other)
        return *this;
    cameraToWorld = other.cameraToWorld;
    focalLengthMm = other.focalLengthMm;
    focusDistanceCm = other.focusDistanceCm;
    exposure = other.exposure;
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        other.retainedPsfGridPath);
    return *this;
}

std::unique_ptr<Sensor> Camera::releaseSensor()
{
    return std::unique_ptr<Sensor>(sensor.release());
}

void Camera::setSensor(std::unique_ptr<Sensor> newSensor)
{
    if (!newSensor)
        throw std::invalid_argument("Camera requires a Sensor");
    if (!*newSensor)
        throw std::invalid_argument("Camera requires a tagged concrete Sensor type");

    nr::synchronizeBeforeManagedMutation("Camera sensor replacement");

    const std::string psfPath = getSensor().getPsfGridPath();
    if (!psfPath.empty())
        std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s", psfPath.c_str());

    sensor.reset(newSensor.release());
    if (sensor->getType() != SensorType::Rectangular
        && retainedPsfGridPath[0] != '\0'
        && sensor->getPsfGridPath().empty())
        sensor->loadPsfGrid(retainedPsfGridPath);
}

PerspectiveCamera::PerspectiveCamera()
    : PerspectiveCamera(std::make_unique<RectangularSensor>()) {}
PerspectiveCamera::PerspectiveCamera(std::unique_ptr<Sensor> ownedSensor)
    : TaggedBase(std::move(ownedSensor)) {}
ThinLensCamera::ThinLensCamera()
    : ThinLensCamera(std::make_unique<RectangularSensor>()) {}
ThinLensCamera::ThinLensCamera(std::unique_ptr<Sensor> ownedSensor)
    : TaggedBase(std::move(ownedSensor)) {}
OrthographicCamera::OrthographicCamera()
    : OrthographicCamera(std::make_unique<RectangularSensor>()) {}
OrthographicCamera::OrthographicCamera(std::unique_ptr<Sensor> ownedSensor)
    : TaggedBase(std::move(ownedSensor)) {}
FisheyeCamera::FisheyeCamera()
    : FisheyeCamera(std::make_unique<RectangularSensor>()) {}
FisheyeCamera::FisheyeCamera(std::unique_ptr<Sensor> ownedSensor)
    : TaggedBase(std::move(ownedSensor)) {}

PerspectiveCamera::PerspectiveCamera(const PerspectiveCamera& other)
    : TaggedBase(other) {}


ThinLensCamera::ThinLensCamera(const ThinLensCamera& other)
    : TaggedBase(other), apertureDiameterMm(other.apertureDiameterMm), bokehBias(other.bokehBias) {}


OrthographicCamera::OrthographicCamera(const OrthographicCamera& other)
    : TaggedBase(other) {}


FisheyeCamera::FisheyeCamera(const FisheyeCamera& other)
    : TaggedBase(other), apertureDiameterMm(other.apertureDiameterMm), bokehBias(other.bokehBias) {}


PerspectiveCamera::~PerspectiveCamera() = default;
ThinLensCamera::~ThinLensCamera() = default;
OrthographicCamera::~OrthographicCamera() = default;
FisheyeCamera::~FisheyeCamera() = default;

float Camera::focalLengthMmForFovDegrees(const float fovDegrees) const
{
    const float halfAngle = glm::radians(std::clamp(fovDegrees, 1.f, 179.f)) * 0.5f;
    return getSensor().filmWidth() / (2.f * std::tan(halfAngle));
}

float Camera::fovDegreesForFocalLengthMm(const float requestedFocalLengthMm) const
{
    const float fovDegrees = 2.f * std::atan(
        getSensor().filmWidth() / (2.f * std::max(0.001f, requestedFocalLengthMm)))
        * (180.f / std::numbers::pi_v<float>);
    return std::clamp(fovDegrees, 1.f, 179.f);
}

void Camera::setFocalLengthMm(const float requestedFocalLengthMm)
{
    nr::synchronizeBeforeManagedMutation("Camera focal length");
    if (ptr()) {
        DispatchCPU([requestedFocalLengthMm](auto* cam) {
            cam->focalLengthMm = std::max(0.001f, requestedFocalLengthMm);
        });
    } else {
        focalLengthMm = std::max(0.001f, requestedFocalLengthMm);
    }
}

void Camera::setFocusDistanceCm(const float requestedFocusDistanceCm)
{
    nr::synchronizeBeforeManagedMutation("Camera focus distance");
    if (ptr()) {
        DispatchCPU([requestedFocusDistanceCm](auto* cam) {
            using CameraType = std::remove_cvref_t<decltype(*cam)>;
            if constexpr (std::is_same_v<CameraType, RealisticCamera>
                || std::is_same_v<CameraType, HybridPsfCamera>)
                cam->setOpticalFocusDistanceCm(requestedFocusDistanceCm);
            else
                cam->focusDistanceCm = std::max(0.1f, requestedFocusDistanceCm);
        });
    } else {
        focusDistanceCm = std::max(0.1f, requestedFocusDistanceCm);
    }
}

void Camera::setExposure(const float requestedExposure)
{
    nr::synchronizeBeforeManagedMutation("Camera exposure");
    if (ptr())
        DispatchCPU([requestedExposure](auto* cam) { cam->exposure = requestedExposure; });
    else
        exposure = requestedExposure;
}

float Camera::getFocusDistanceCm() const
{
    if (ptr())
        return DispatchCPU([](const auto* cam) { return cam->focusDistanceCm; });
    return focusDistanceCm;
}

void Camera::prepareForRender()
{
    if (!ptr())
        return;
    DispatchCPU([](auto* camera) {
        using CameraType = std::remove_cvref_t<decltype(*camera)>;
        if constexpr (std::is_same_v<CameraType, RealisticCamera>
            || std::is_same_v<CameraType, HybridPsfCamera>)
            camera->prepareOptics();
    });
}

float Camera::getFocalLengthMm() const
{
    if (ptr())
        return DispatchCPU([](const auto* cam) { return cam->focalLengthMm; });
    return focalLengthMm;
}

void Camera::setCameraToWorld(const glm::mat4& m)
{
    nr::synchronizeBeforeManagedMutation("Camera transform");
    if (ptr())
        DispatchCPU([&m](auto* cam) { cam->cameraToWorld = m; });
    else
        cameraToWorld = m;
}

Camera Camera::cloneBaseState() const
{
    const Camera* source = ptr()
        ? DispatchCPU([](const auto* cam) { return static_cast<const Camera*>(cam); })
        : this;

    Camera state;
    state.cameraToWorld = source->cameraToWorld;
    state.focalLengthMm = source->focalLengthMm;
    state.focusDistanceCm = source->focusDistanceCm;
    state.exposure = source->exposure;
    std::snprintf(state.retainedPsfGridPath, sizeof(state.retainedPsfGridPath), "%s",
        source->retainedPsfGridPath);
    return state;
}
