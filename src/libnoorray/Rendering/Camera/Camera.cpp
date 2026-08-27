#include "Rendering/Camera/Camera.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <type_traits>

#include "Backend/Host/MutationBarrier.h"

Camera::~Camera() = default;

Camera::Camera(std::unique_ptr<Sensor> ownedSensor)
    : sensor(ownedSensor.release())
{
    if (!sensor)
        throw std::invalid_argument("Camera requires a Sensor");
    if (!*sensor)
        throw std::invalid_argument("Camera requires a tagged concrete Sensor type");
}

Camera::Camera(const Camera& other)
    : cameraToWorld(other.cameraToWorld)
    , focalLengthMm(other.focalLengthMm)
    , focusDistanceCm(other.focusDistanceCm), exposure(other.exposure)
{
    const Sensor& source = other.getSensor();
    if (const auto* rectangular = dynamic_cast<const RectangularSensor*>(&source))
        sensor = std::make_unique<RectangularSensor>(*rectangular);
    else
        sensor = std::make_unique<Sensor>(source);
}

Camera& Camera::operator=(const Camera& other)
{
    if (this == &other)
        return *this;
    cameraToWorld = other.cameraToWorld;
    focalLengthMm = other.focalLengthMm;
    focusDistanceCm = other.focusDistanceCm;
    exposure = other.exposure;
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

    sensor.reset(newSensor.release());
}

PerspectiveCamera::PerspectiveCamera()
    : PerspectiveCamera(std::make_unique<RectangularSensor>()) {}
PerspectiveCamera::PerspectiveCamera(std::unique_ptr<Sensor> ownedSensor)
    : Camera(std::move(ownedSensor)) {}
ThinLensCamera::ThinLensCamera()
    : ThinLensCamera(std::make_unique<RectangularSensor>()) {}
ThinLensCamera::ThinLensCamera(std::unique_ptr<Sensor> ownedSensor)
    : Camera(std::move(ownedSensor)) {}
OrthographicCamera::OrthographicCamera()
    : OrthographicCamera(std::make_unique<RectangularSensor>()) {}
OrthographicCamera::OrthographicCamera(std::unique_ptr<Sensor> ownedSensor)
    : Camera(std::move(ownedSensor)) {}
FisheyeCamera::FisheyeCamera()
    : FisheyeCamera(std::make_unique<RectangularSensor>()) {}
FisheyeCamera::FisheyeCamera(std::unique_ptr<Sensor> ownedSensor)
    : Camera(std::move(ownedSensor)) {}

PerspectiveCamera::PerspectiveCamera(const PerspectiveCamera& other)
    : Camera(other) {}


ThinLensCamera::ThinLensCamera(const ThinLensCamera& other)
    : Camera(other), apertureDiameterMm(other.apertureDiameterMm), bokehBias(other.bokehBias) {}


OrthographicCamera::OrthographicCamera(const OrthographicCamera& other)
    : Camera(other) {}


FisheyeCamera::FisheyeCamera(const FisheyeCamera& other)
    : Camera(other), apertureDiameterMm(other.apertureDiameterMm), bokehBias(other.bokehBias) {}


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
    if (auto* realistic = dynamic_cast<RealisticCamera*>(this))
        realistic->setOpticalFocusDistanceCm(requestedFocusDistanceCm);
    else
        focusDistanceCm = std::max(0.1f, requestedFocusDistanceCm);
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
    if (auto* realistic = dynamic_cast<RealisticCamera*>(this))
        realistic->prepareOptics();
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
    const Camera* source = this;

    Camera state;
    state.cameraToWorld = source->cameraToWorld;
    state.focalLengthMm = source->focalLengthMm;
    state.focusDistanceCm = source->focusDistanceCm;
    state.exposure = source->exposure;
    return state;
}
