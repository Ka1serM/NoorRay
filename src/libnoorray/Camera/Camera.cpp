#include "Camera/Camera.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <type_traits>

#include "CUDA/ManagedMemory.h"
#include "CUDA/rstd/Allocator.h"
#include "UI/ImGuiManager.h"

Camera::~Camera() = default;

Camera::Camera(std::unique_ptr<Sensor> ownedSensor)
    : sensorOwner(ownedSensor.release())
{
    if (!sensorOwner)
        throw std::invalid_argument("Camera requires a Sensor");
    tagSensor();
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        sensorOwner->getPsfGridPath().c_str());
}

Camera::Camera(const Camera& other)
    : TaggedCamera(other), sensor(other.sensor), cameraToWorld(other.cameraToWorld)
    , fieldOfView(other.fieldOfView), focalLengthMm(other.focalLengthMm)
    , focusDistance(other.focusDistance)
{
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        other.retainedPsfGridPath);
}

Camera& Camera::operator=(const Camera& other)
{
    if (this == &other)
        return *this;
    static_cast<TaggedCamera&>(*this) = static_cast<const TaggedCamera&>(other);
    if (sensorOwner) {
        sensor.copyPhysicalFrom(*sensorOwner);
        static_cast<TaggedSensor&>(sensor) = static_cast<const TaggedSensor&>(*sensorOwner);
    } else {
        sensor = other.sensor;
    }
    cameraToWorld = other.cameraToWorld;
    fieldOfView = other.fieldOfView;
    focalLengthMm = other.focalLengthMm;
    focusDistance = other.focusDistance;
    std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s",
        other.retainedPsfGridPath);
    return *this;
}

void Camera::tagSensor()
{
    TaggedSensor tagged;
    if (auto* concrete = dynamic_cast<ScatterPsfSensor*>(sensorOwner.get()))
        tagged = TaggedSensor(concrete);
    else if (auto* concrete = dynamic_cast<GatherPsfSensor*>(sensorOwner.get()))
        tagged = TaggedSensor(concrete);
    else if (auto* concrete = dynamic_cast<RectangularSensor*>(sensorOwner.get()))
        tagged = TaggedSensor(concrete);
    else
        throw std::invalid_argument("Camera requires a concrete Sensor type");

    static_cast<TaggedSensor&>(*sensorOwner) = tagged;
    sensor.copyPhysicalFrom(*sensorOwner);
    static_cast<TaggedSensor&>(sensor) = tagged;
}

std::unique_ptr<Sensor> Camera::releaseSensor()
{
    return std::unique_ptr<Sensor>(sensorOwner.release());
}

void Camera::setSensor(std::unique_ptr<Sensor> newSensor)
{
    if (!newSensor)
        throw std::invalid_argument("Camera requires a Sensor");

    const std::string psfPath = sensorOwner
        ? sensorOwner->getPsfGridPath() : std::string{};
    if (!psfPath.empty())
        std::snprintf(retainedPsfGridPath, sizeof(retainedPsfGridPath), "%s", psfPath.c_str());

    sensorOwner.reset(newSensor.release());
    tagSensor();
    if (sensorOwner->getType() != SensorType::Rectangular
        && retainedPsfGridPath[0] != '\0'
        && sensorOwner->getPsfGridPath().empty())
        sensorOwner->loadPsfGrid(retainedPsfGridPath);
}

std::unique_ptr<Camera> Camera::create(
    const CameraProjectionType projectionType, std::unique_ptr<Sensor> ownedSensor)
{
    const auto create = [&]<typename CameraType>() -> std::unique_ptr<Camera> {
        auto camera = ownedSensor
            ? std::make_unique<CameraType>(std::move(ownedSensor))
            : std::make_unique<CameraType>();
        static_cast<TaggedCamera&>(*camera) = TaggedCamera(camera.get());
        return camera;
    };
    switch (projectionType) {
    case CameraProjectionType::ThinLens: return create.template operator()<ThinLensCamera>();
    case CameraProjectionType::Orthographic: return create.template operator()<OrthographicCamera>();
    case CameraProjectionType::Fisheye: return create.template operator()<FisheyeCamera>();
    case CameraProjectionType::Realistic: return create.template operator()<RealisticCamera>();
    case CameraProjectionType::HybridPsf: return create.template operator()<RossPsfCamera>();
    case CameraProjectionType::Perspective:
    default: return create.template operator()<PerspectiveCamera>();
    }
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
    : Camera(other.cloneBaseState())
{
    sensor = Sensor(nullptr);
    sensor.cloneConcreteFrom(other.getSensor());
}


ThinLensCamera::ThinLensCamera(const ThinLensCamera& other)
    : Camera(other.cloneBaseState()), fStop(other.fStop), bokehBias(other.bokehBias)
{
    sensor = Sensor(nullptr);
    sensor.cloneConcreteFrom(other.getSensor());
}


OrthographicCamera::OrthographicCamera(const OrthographicCamera& other)
    : Camera(other.cloneBaseState())
{
    sensor = Sensor(nullptr);
    sensor.cloneConcreteFrom(other.getSensor());
}


FisheyeCamera::FisheyeCamera(const FisheyeCamera& other)
    : Camera(other.cloneBaseState()), fStop(other.fStop), bokehBias(other.bokehBias)
{
    sensor = Sensor(nullptr);
    sensor.cloneConcreteFrom(other.getSensor());
}


PerspectiveCamera::~PerspectiveCamera() { sensor.freeConcrete(); }
ThinLensCamera::~ThinLensCamera() { sensor.freeConcrete(); }
OrthographicCamera::~OrthographicCamera() { sensor.freeConcrete(); }
FisheyeCamera::~FisheyeCamera() { sensor.freeConcrete(); }

float Camera::focalLengthForFov(const float fovDegrees) const
{
    const float halfAngle = glm::radians(std::clamp(fovDegrees, 1.f, 179.f)) * 0.5f;
    return sensor.width() / (2.f * std::tan(halfAngle));
}

float Camera::fovForFocalLength(const float focalLength) const
{
    const float fov = 2.f * std::atan(sensor.width() / (2.f * std::max(0.001f, focalLength)))
        * (180.f / std::numbers::pi_v<float>);
    return std::clamp(fov, 1.f, 179.f);
}

void Camera::setFocalLength(const float focalLength)
{
    nr::synchronizeBeforeManagedMutation("Camera focal length");
    if (ptr()) {
        DispatchCPU([focalLength](auto* cam) {
            cam->focalLengthMm = std::max(0.001f, focalLength);
            cam->fieldOfView = cam->fovForFocalLength(cam->focalLengthMm);
        });
    } else {
        focalLengthMm = std::max(0.001f, focalLength);
        fieldOfView = fovForFocalLength(focalLengthMm);
    }
}

void Camera::setFocusDistance(const float v)
{
    nr::synchronizeBeforeManagedMutation("Camera focus distance");
    if (ptr()) {
        DispatchCPU([v](auto* cam) {
            using CameraType = std::remove_cvref_t<decltype(*cam)>;
            if constexpr (std::is_same_v<CameraType, RealisticCamera>
                || std::is_same_v<CameraType, RossPsfCamera>)
                cam->setOpticalFocusDistance(v);
            else
                cam->focusDistance = std::max(0.001f, v);
        });
    } else {
        focusDistance = std::max(0.001f, v);
    }
}

float Camera::getFocusDistance() const
{
    if (ptr())
        return DispatchCPU([](const auto* cam) { return cam->focusDistance; });
    return focusDistance;
}

void Camera::prepareForRender()
{
    if (!ptr())
        return;
    DispatchCPU([](auto* camera) {
        using CameraType = std::remove_cvref_t<decltype(*camera)>;
        if constexpr (std::is_same_v<CameraType, RealisticCamera>
            || std::is_same_v<CameraType, RossPsfCamera>)
            camera->prepareOptics();
    });
}

Sensor& Camera::getSensor()
{
    if (sensorOwner)
        return *sensorOwner;
    if (ptr())
        return DispatchCPU([](auto* cam) -> Sensor& { return cam->sensor; });
    return sensor;
}

const Sensor& Camera::getSensor() const
{
    if (sensorOwner)
        return *sensorOwner;
    if (ptr())
        return DispatchCPU([](const auto* cam) -> const Sensor& { return cam->sensor; });
    return sensor;
}

float Camera::getFocalLength() const
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
    Camera state = ptr()
        ? DispatchCPU([](const auto* cam) { return static_cast<const Camera&>(*cam); })
        : *this;
    static_cast<TaggedCamera&>(state) = TaggedCamera(nullptr);
    static_cast<TaggedSensor&>(state.sensor) = TaggedSensor(nullptr);
    return state;
}


bool Camera::renderUi()
{
    bool changed = false;
    ImGuiManager::dragFloatRow("Field of View", fieldOfView, 0.1f, 1.f, 179.f, [&](float value) {
        nr::synchronizeBeforeManagedMutation("Camera field of view");
        fieldOfView = std::clamp(value, 1.f, 179.f);
        focalLengthMm = focalLengthForFov(fieldOfView);
        changed = true;
    });
    ImGuiManager::dragFloatRow("Focal Length", focalLengthMm, 0.1f, 0.001f, 500.f, [&](float value) {
        setFocalLength(value);
        changed = true;
    });
    return changed;
}
