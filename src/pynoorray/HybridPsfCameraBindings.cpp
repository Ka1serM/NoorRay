#include "Bindings.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "Camera/RossPsfCamera.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindHybridPsfCamera(nb::module_& module)
{
    nb::class_<RossPsfCamera, Camera>(module, "HybridPsfCamera")
        .def(nb::new_([] { return new RossPsfCamera(); }))
        .def(nb::new_([](std::unique_ptr<Sensor> sensor) {
            return new RossPsfCamera(std::move(sensor));
        }), "sensor"_a)
        .def("load",
            nb::overload_cast<std::string, const std::vector<std::string>&, std::string>(
                &RossPsfCamera::load),
            "lens_path"_a, "glass_catalog_paths"_a, "ray_lut_path"_a = "")
        .def_prop_rw("aperture_diameter",
            [](const RossPsfCamera& camera) { return camera.apertureDiameterMm; },
            &RossPsfCamera::setApertureDiameter)
        .def_rw("ray_lut_step_size", &RossPsfCamera::rayLutStepSize)
        .def_rw("samples_per_dimension", &RossPsfCamera::samplesPerDimension);
}
