#include "Bindings.h"

#include "Materials/Material.h"

namespace nb = nanobind;

// Materials are authored through MaterialX and compiled by the renderer, so
// Python only inspects the compiled result.
void bindMaterial(nb::module_& module)
{
    nb::class_<Material>(module, "Material")
        .def_prop_ro("has_program", &Material::hasProgram)
        .def_prop_ro("bytecode_length", [](const Material& material) {
            return static_cast<std::uint32_t>(material.program.bytecode.size());
        })
        .def_prop_ro("stack_size", [](const Material& material) {
            return material.program.stackSize;
        })
        .def_ro("shadow_opaque", &Material::shadowOpaque)
        .def_ro("may_emit", &Material::mayEmit);
}
