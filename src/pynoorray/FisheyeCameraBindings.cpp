#include "Bindings.h"

#include <nanobind/stl/unique_ptr.h>

#include "Camera/FisheyeCamera.h"

namespace nb = nanobind;

void bindFisheyeCamera(nb::module_& module)
{
    nb::class_<FisheyeCamera, Camera>(module, "FisheyeCamera")
        .def(nb::new_([] { return new FisheyeCamera(); }))
        .def(nb::new_([](std::unique_ptr<Sensor> sensor) {
            return new FisheyeCamera(std::move(sensor));
        }))
        .def_rw("aperture_diameter_mm", &FisheyeCamera::apertureDiameterMm)
        .def_rw("bokeh_bias", &FisheyeCamera::bokehBias);
}
