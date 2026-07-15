#include "Bindings.h"

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "Scene/SceneObject.h"

namespace nb = nanobind;

void bindSceneObject(nb::module_& module)
{
    nb::class_<SceneObject>(module, "SceneObject")
        .def_prop_ro("id", &SceneObject::getId)
        .def_prop_ro("name", &SceneObject::getName)
        .def_prop_rw("visible", &SceneObject::isVisible, &SceneObject::setVisible)
        .def_prop_rw("position", &SceneObject::getPosition, &SceneObject::setPosition)
        .def_prop_rw("rotation", &SceneObject::getRotationEuler, &SceneObject::setRotationEuler)
        .def_prop_rw("scale", &SceneObject::getScale, &SceneObject::setScale)
        .def_prop_rw("transform", &SceneObject::getTransform, &SceneObject::setLocalTransform)
        .def_prop_ro("children", &SceneObject::getChildren)
        .def_prop_ro("parent", &SceneObject::getParentPtr);
}
