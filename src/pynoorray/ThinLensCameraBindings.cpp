#include "Bindings.h"

#include <nanobind/stl/unique_ptr.h>

#include "Camera/ThinLensCamera.h"

namespace nb = nanobind;

void bindThinLensCamera(nb::module_& module)
{
    nb::class_<ThinLensCamera, Camera>(module, "ThinLensCamera")
        .def(nb::new_([] { return new ThinLensCamera(); }))
        .def(nb::new_([](std::unique_ptr<Sensor> sensor) {
            return new ThinLensCamera(std::move(sensor));
        }))
        .def_rw("f_stop", &ThinLensCamera::fStop)
        .def_rw("bokeh_bias", &ThinLensCamera::bokehBias);
}
