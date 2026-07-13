#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "Vulkan/Context.h"
#include "Vulkan/Texture.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindTexture(nb::module_& module)
{
    nb::class_<Texture>(module, "Texture")
        .def(nb::new_([](Context& context, const std::string& path) {
            return new Texture(context, path);
        }), "context"_a, "path"_a)
        .def_prop_ro("name", &Texture::getName)
        .def_prop_ro("width", &Texture::getWidth)
        .def_prop_ro("height", &Texture::getHeight);
}
