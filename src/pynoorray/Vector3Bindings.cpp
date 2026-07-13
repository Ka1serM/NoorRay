#include "Bindings.h"

#include <glm/vec3.hpp>
#include <nanobind/operators.h>

namespace nb = nanobind;
using namespace nb::literals;

void bindVector3(nb::module_& module)
{
    nb::class_<glm::vec3>(module, "Vector3")
        .def(nb::init<>())
        .def(nb::init<float>(), "value"_a)
        .def(nb::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
        .def_rw("x", &glm::vec3::x)
        .def_rw("y", &glm::vec3::y)
        .def_rw("z", &glm::vec3::z)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * float())
        .def(nb::self / float());
}
