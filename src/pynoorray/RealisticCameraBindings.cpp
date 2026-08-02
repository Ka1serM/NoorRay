#include "Bindings.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "Rendering/Camera/RealisticCamera.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindRealisticCamera(nb::module_& module)
{
    nb::class_<RealisticCamera, Camera>(module, "RealisticCamera")
        .def(nb::new_([] { return new RealisticCamera(); }))
        .def(nb::new_([](std::unique_ptr<Sensor> sensor) {
            return new RealisticCamera(std::move(sensor));
        }), "sensor"_a)
        .def("load",
            nb::overload_cast<std::string, const std::vector<std::string>&>(
                &RealisticCamera::load),
            "lens_path"_a, "glass_catalog_paths"_a)
        .def_prop_rw("aperture_diameter_mm",
            [](const RealisticCamera& camera) { return camera.apertureDiameterMm; },
            &RealisticCamera::setApertureDiameterMm);
}
