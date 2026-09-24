#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "IO/TextureReader.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindTexture(nb::module_& module)
{
    nb::class_<Texture>(module, "Texture")
        .def("__init__", [](Texture* texture, const std::string& path) {
            new (texture) Texture(TextureReader::read(path));
        }, "path"_a)
        .def_prop_ro("name", &Texture::getName)
        .def_prop_ro("width", &Texture::getWidth)
        .def_prop_ro("height", &Texture::getHeight);
}
