#include "Bindings.h"

#include <nanobind/stl/unique_ptr.h>

#include "Camera/OrthographicCamera.h"

namespace nb = nanobind;

void bindOrthographicCamera(nb::module_& module)
{
    nb::class_<OrthographicCamera, Camera>(module, "OrthographicCamera")
        .def(nb::new_([] { return new OrthographicCamera(); }))
        .def(nb::new_([](std::unique_ptr<Sensor> sensor) {
            return new OrthographicCamera(std::move(sensor));
        }));
}
