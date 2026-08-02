#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "Scene/Resources/Texture.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindTexture(nb::module_& module)
{
    nb::class_<Texture>(module, "Texture")
        .def(nb::init<const std::string&>(), "path"_a)
        .def_prop_ro("name", &Texture::getName)
        .def_prop_ro("width", &Texture::getWidth)
        .def_prop_ro("height", &Texture::getHeight);
}
