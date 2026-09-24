#include "Bindings.h"

#include "Materials/Material.h"

namespace nb = nanobind;

// Materials are authored through MaterialX and compiled by the renderer, so
// Python only inspects the compiled result.
void bindMaterial(nb::module_& module)
{
    nb::class_<Material>(module, "Material")
        .def_ro("compiled", &Material::compiled);
}
