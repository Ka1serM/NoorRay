#include "Bindings.h"

#include "Materials/Shading/Material.h"

namespace nb = nanobind;

void bindMaterial(nb::module_& module)
{
    nb::class_<Material>(module, "Material")
        .def(nb::init<>())
        .def_rw("svm_bytecode_offset", &Material::svmBytecodeOffset)
        .def_rw("svm_bytecode_length", &Material::svmBytecodeLength)
        .def_rw("svm_texture_offset", &Material::svmTextureOffset)
        .def_rw("svm_texture_count", &Material::svmTextureCount);
}
