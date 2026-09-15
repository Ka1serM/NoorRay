#include "Bindings.h"

#include <string>

#include <nanobind/operators.h>
#include <nanobind/stl/string.h>

#include "Scene/Handle.h"

namespace nb = nanobind;

namespace
{
// Handles are opaque on the Python side: they compare, hash and print, but the
// index is exposed read-only because nothing outside the engine should be
// making one up.
void bindHandle(nb::module_& module, const char* name)
{
    using HandleType = SceneObjectHandle;
    nb::class_<HandleType>(module, name)
        .def(nb::init<>())
        .def_prop_ro("index", &HandleType::index)
        .def_prop_ro("generation", &HandleType::generation)
        .def_prop_ro("valid", &HandleType::isValid)
        .def("__bool__", &HandleType::isValid)
        .def(nb::self == nb::self)
        .def("__hash__", [](const HandleType& handle) {
            return std::hash<HandleType>{}(handle);
        })
        .def("__repr__", [name](const HandleType& handle) {
            return std::string(name) + "(" + std::to_string(handle.index())
                + ", " + std::to_string(handle.generation()) + ")";
        });
}

}

void bindHandles(nb::module_& module)
{
    bindHandle(module, "SceneObjectHandle");
}
