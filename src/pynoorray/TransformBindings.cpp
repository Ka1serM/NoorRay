#include "Bindings.h"

#include "Geometry/Mesh/Transform.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindTransform(nb::module_& module)
{
    nb::class_<Transform>(module, "Transform")
        .def(nb::init<>())
        .def(nb::init<glm::vec3>(), "position"_a)
        .def(nb::init<glm::vec3, glm::vec3, glm::vec3>(),
            "position"_a, "rotation"_a, "scale"_a)
        .def_prop_rw("position", &Transform::getPosition, &Transform::setPosition)
        .def_prop_rw("rotation", &Transform::getRotationEuler, &Transform::setRotationEuler)
        .def_prop_rw("scale", &Transform::getScale, &Transform::setScale);
}
