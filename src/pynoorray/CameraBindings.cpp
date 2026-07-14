#include "Bindings.h"

#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/pair.h>

#include "Camera/Camera.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindCamera(nb::module_& module)
{
    nb::enum_<CameraProjectionType>(module, "CameraProjectionType")
        .value("PERSPECTIVE", CameraProjectionType::Perspective)
        .value("ORTHOGRAPHIC", CameraProjectionType::Orthographic)
        .value("FISHEYE", CameraProjectionType::Fisheye)
        .value("THIN_LENS", CameraProjectionType::ThinLens)
        .value("REALISTIC", CameraProjectionType::Realistic)
        .value("ROSS_PSF", CameraProjectionType::HybridPsf)
        .value("HYBRID_PSF", CameraProjectionType::HybridPsf);

    nb::class_<Camera>(module, "Camera")
        .def_prop_rw("sensor_resolution",
            [](const Camera& camera) {
                const Sensor& sensor = camera.getSensor();
                return std::pair(sensor.resolutionX(), sensor.resolutionY());
            },
            [](Camera& camera, const std::pair<uint32_t, uint32_t>& resolution) {
                camera.getSensor().setResolution(resolution.first, resolution.second);
            })
        .def_prop_rw("focal_length_mm", &Camera::getFocalLengthMm, &Camera::setFocalLengthMm)
        .def_prop_rw("focus_distance_cm", &Camera::getFocusDistanceCm, &Camera::setFocusDistanceCm)
        .def_prop_ro("sensor", nb::overload_cast<>(&Camera::getSensor), nb::rv_policy::reference_internal)
        .def("set_sensor", &Camera::setSensor, "sensor"_a);
}
