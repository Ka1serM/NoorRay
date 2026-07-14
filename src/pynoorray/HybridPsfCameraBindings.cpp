#include "Bindings.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "Camera/HybridPsfCamera.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindHybridPsfCamera(nb::module_& module)
{
    nb::class_<HybridPsfCamera, Camera>(module, "HybridPsfCamera")
        .def(nb::new_([] { return new HybridPsfCamera(); }))
        .def(nb::new_([](std::unique_ptr<Sensor> sensor) {
            return new HybridPsfCamera(std::move(sensor));
        }), "sensor"_a)
        .def("load",
            nb::overload_cast<std::string, const std::vector<std::string>&, std::string>(
                &HybridPsfCamera::load),
            "lens_path"_a, "glass_catalog_paths"_a, "ray_lut_path"_a = "")
        .def_prop_rw("aperture_diameter_mm",
            [](const HybridPsfCamera& camera) { return camera.apertureDiameterMm; },
            &HybridPsfCamera::setApertureDiameterMm)
        .def_rw("ray_lut_step_size", &HybridPsfCamera::rayLutStepSize)
        .def_rw("samples_per_dimension", &HybridPsfCamera::samplesPerDimension);
}
