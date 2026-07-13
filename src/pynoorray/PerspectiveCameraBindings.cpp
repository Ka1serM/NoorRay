#include "Bindings.h"

#include <nanobind/stl/unique_ptr.h>

#include "Camera/PerspectiveCamera.h"

namespace nb = nanobind;

void bindPerspectiveCamera(nb::module_& module)
{
    nb::class_<PerspectiveCamera, Camera>(module, "PerspectiveCamera")
        .def(nb::new_([] { return new PerspectiveCamera(); }))
        .def(nb::new_([](std::unique_ptr<Sensor> sensor) {
            return new PerspectiveCamera(std::move(sensor));
        }));
}
