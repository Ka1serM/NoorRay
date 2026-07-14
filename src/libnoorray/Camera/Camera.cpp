#include "Camera/Camera.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <type_traits>

#include "CUDA/ManagedMemory.h"
#include "UI/ImGuiManager.h"

Camera::~Camera() = default;

Camera::Camera(std::unique_ptr<Sensor> ownedSensor)
    : sensor(ownedSensor.release())
{
    if (!sensor)
        throw std::invalid_argument("Camera requires a Sensor");
    tagSensor();
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        sensor->getPsfGridPath().c_str());
}

Camera::Camera(const Camera& other)
    : TaggedCamera(other), cameraToWorld(other.cameraToWorld)
    , fieldOfViewDegrees(other.fieldOfViewDegrees), focalLengthMm(other.focalLengthMm)
    , focusDistanceCm(other.focusDistanceCm)
{
    const Sensor& source = other.getSensor();
    source.DispatchCPU([this, &source](const auto* concrete) {
        using SensorType = std::remove_cvref_t<decltype(*concrete)>;
        sensor.reset(new SensorType(source));
    });
    tagSensor();
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        other.retainedPsfGridPath);
}

Camera& Camera::operator=(const Camera& other)
{
    if (this == &other)
        return *this;
    cameraToWorld = other.cameraToWorld;
    fieldOfViewDegrees = other.fieldOfViewDegrees;
    focalLengthMm = other.focalLengthMm;
    focusDistanceCm = other.focusDistanceCm;
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        other.retainedPsfGridPath);
    return *this;
}

void Camera::tagSensor()
{
    TaggedSensor tagged;
    if (auto* concrete = dynamic_cast<ScatterPsfSensor*>(sensor.get()))
        tagged = TaggedSensor(concrete);
    else if (auto* concrete = dynamic_cast<GatherPsfSensor*>(sensor.get()))
        tagged = TaggedSensor(concrete);
    else if (auto* concrete = dynamic_cast<RectangularSensor*>(sensor.get()))
        tagged = TaggedSensor(concrete);
    else
        throw std::invalid_argument("Camera requires a concrete Sensor type");

    static_cast<TaggedSensor&>(*sensor) = tagged;
}

std::unique_ptr<Sensor> Camera::releaseSensor()
{
    return std::unique_ptr<Sensor>(sensor.release());
}

void Camera::setSensor(std::unique_ptr<Sensor> newSensor)
{
    if (!newSensor)
        throw std::invalid_argument("Camera requires a Sensor");

    nr::synchronizeBeforeManagedMutation("Camera sensor replacement");

    const std::string psfPath = getSensor().getPsfGridPath();
    if (!psfPath.empty())
        std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s", psfPath.c_str());

    sensor.reset(newSensor.release());
    tagSensor();
    if (sensor->getType() != SensorType::Rectangular
        && retainedPsfGridPath[0] != '\0'
        && sensor->getPsfGridPath().empty())
        sensor->loadPsfGrid(retainedPsfGridPath);
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
    return getSensor().width() / (2.f * std::tan(halfAngle));
}

float Camera::fovDegreesForFocalLengthMm(const float requestedFocalLengthMm) const
{
    const float fovDegrees = 2.f * std::atan(
        getSensor().width() / (2.f * std::max(0.001f, requestedFocalLengthMm)))
        * (180.f / std::numbers::pi_v<float>);
    return std::clamp(fovDegrees, 1.f, 179.f);
}

void Camera::setFocalLengthMm(const float requestedFocalLengthMm)
{
    nr::synchronizeBeforeManagedMutation("Camera focal length");
    if (ptr()) {
        DispatchCPU([requestedFocalLengthMm](auto* cam) {
            cam->focalLengthMm = std::max(0.001f, requestedFocalLengthMm);
            cam->fieldOfViewDegrees = cam->fovDegreesForFocalLengthMm(cam->focalLengthMm);
        });
    } else {
        focalLengthMm = std::max(0.001f, requestedFocalLengthMm);
        fieldOfViewDegrees = fovDegreesForFocalLengthMm(focalLengthMm);
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
    state.fieldOfViewDegrees = source->fieldOfViewDegrees;
    state.focalLengthMm = source->focalLengthMm;
    state.focusDistanceCm = source->focusDistanceCm;
    std::snprintf(state.retainedPsfGridPath, sizeof(state.retainedPsfGridPath), "%s",
        source->retainedPsfGridPath);
    return state;
}


bool Camera::renderUi()
{
    bool changed = false;
    ImGuiManager::dragFloatRow("Field of View (degrees)", fieldOfViewDegrees, 0.1f, 1.f, 179.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Camera field of view");
        fieldOfViewDegrees = std::clamp(value, 1.f, 179.f);
        focalLengthMm = focalLengthMmForFovDegrees(fieldOfViewDegrees);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focal Length (mm)", focalLengthMm, 0.1f, 0.001f, 500.f, [&](float value) {
        setFocalLengthMm(value);
        changed = true;
    });
    return changed;
}
