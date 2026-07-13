#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "Camera/Sensor.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindScatterPsfSensor(nb::module_& module)
{
    nb::class_<ScatterPsfSensor, RectangularSensor>(module, "ScatterPsfSensor")
        .def(nb::new_([] { return new ScatterPsfSensor(); }))
        .def("load_psf", [](ScatterPsfSensor& sensor, const std::string& path) {
            sensor.psfGridPath = path;
            sensor.loadPsfGrid();
        }, "path"_a);
}
