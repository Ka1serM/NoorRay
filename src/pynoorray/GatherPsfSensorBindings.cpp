#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "Rendering/Camera/Sensor.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindGatherPsfSensor(nb::module_& module)
{
    nb::class_<GatherPsfSensor, RectangularSensor>(module, "GatherPsfSensor")
        .def(nb::new_([] { return new GatherPsfSensor(); }))
        .def("load_psf", [](GatherPsfSensor& sensor, const std::string& path) {
            sensor.psfGridPath = path;
            sensor.loadPsfGrid();
        }, "path"_a);
}
