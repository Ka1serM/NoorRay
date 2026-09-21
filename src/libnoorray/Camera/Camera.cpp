#include "Camera/Camera.h"
#include "Camera/CameraInstance.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <type_traits>

Camera::~Camera() = default;

Camera::Camera()
{
    updateData();
}

Camera::Camera(std::unique_ptr<Sensor> ownedSensor)
    : sensor(ownedSensor.release())
{
    if (!sensor)
        throw std::invalid_argument("Camera requires a Sensor");
    if (!*sensor)
        throw std::invalid_argument("Camera requires a tagged concrete Sensor type");
    updateData();
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
    sensor->owner_ = this;
    data = other.data;
}

Camera& Camera::operator=(const Camera& other)
{
    if (this == &other)
        return *this;
    cameraToWorld = other.cameraToWorld;
    focalLengthMm = other.focalLengthMm;
    focusDistanceCm = other.focusDistanceCm;
    exposure = other.exposure;
    data = other.data;
    notifyChanged();
    return *this;
}

void Camera::notifyChanged()
{
    if (owner_) owner_->markDirty();
}

void Sensor::notifyChanged()
{
    if (owner_) owner_->updateData();
}

std::unique_ptr<Sensor> Camera::releaseSensor()
{
    if (sensor) sensor->owner_ = nullptr;
    return std::unique_ptr<Sensor>(sensor.release());
}

void Camera::setSensor(std::unique_ptr<Sensor> newSensor)
{
    if (!newSensor)
        throw std::invalid_argument("Camera requires a Sensor");
    if (!*newSensor)
        throw std::invalid_argument("Camera requires a tagged concrete Sensor type");

    sensor.reset(newSensor.release());
    updateData();
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
    focalLengthMm = std::max(0.001f, requestedFocalLengthMm);
    updateData();
}

void Camera::setFocusDistanceCm(const float requestedFocusDistanceCm)
{
    if (auto* realistic = dynamic_cast<RealisticCamera*>(this))
        realistic->setOpticalFocusDistanceCm(requestedFocusDistanceCm);
    else
        focusDistanceCm = std::max(0.1f, requestedFocusDistanceCm);
    updateData();
}

void Camera::setExposure(const float requestedExposure)
{
    exposure = requestedExposure;
    updateData();
}

float Camera::getExposure() const
{
    return exposure;
}

float Camera::getFocusDistanceCm() const
{
    return focusDistanceCm;
}

void Camera::prepareForRender()
{
    if (auto* realistic = dynamic_cast<RealisticCamera*>(this))
        realistic->prepareOptics();
    updateData();
}

float Camera::getFocalLengthMm() const
{
    return focalLengthMm;
}

void Camera::setCameraToWorld(const glm::mat4& m)
{
    cameraToWorld = m;
    updateData();
}

void Camera::setProjectionType(const CameraProjectionType projection)
{
    data.projection = static_cast<std::uint32_t>(projection);
    notifyChanged();
}

void Camera::setApertureDiameterMm(const float value)
{
    data.apertureDiameterMm = std::max(0.0f, value);
    notifyChanged();
}

void Camera::updateData()
{
    if (sensor) sensor->owner_ = this;
    for (uint32_t row = 0; row < 4; ++row)
        for (uint32_t column = 0; column < 4; ++column)
            data.cameraToWorld[row * 4u + column] = cameraToWorld[column][row];
    if (sensor) {
        data.sensorWidthMm = sensor->filmWidth();
        data.sensorHeightMm = sensor->filmHeight();
        data.sensorOrigin = static_cast<std::uint32_t>(sensor->origin());
    }
    data.focalLengthMm = focalLengthMm;
    data.focusDistanceCm = focusDistanceCm;
    data.exposure = exposure;
    notifyChanged();
}

Camera Camera::cloneBaseState() const
{
    const Camera* source = this;

    Camera state;
    state.cameraToWorld = source->cameraToWorld;
    state.focalLengthMm = source->focalLengthMm;
    state.focusDistanceCm = source->focusDistanceCm;
    state.exposure = source->exposure;
    state.data = source->data;
    return state;
}
