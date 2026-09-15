#include "Bindings.h"

#include "Camera/Sensor.h"

namespace nb = nanobind;

void bindRectangularSensor(nb::module_& module)
{
    nb::class_<RectangularSensor, Sensor>(module, "RectangularSensor")
        .def(nb::new_([] { return new RectangularSensor(); }));
}
