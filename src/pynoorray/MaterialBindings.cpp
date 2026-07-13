#include "Bindings.h"

#include "Mesh/Material.h"

namespace nb = nanobind;

void bindMaterial(nb::module_& module)
{
    nb::class_<Material>(module, "Material")
        .def(nb::init<>())
        .def_rw("albedo", &Material::albedo)
        .def_rw("specular", &Material::specular)
        .def_rw("metallic", &Material::metallic)
        .def_rw("roughness", &Material::roughness)
        .def_rw("sellmeier", &Material::sellmeier)
        .def_rw("transmission_color", &Material::transmissionColor)
        .def_rw("transmission", &Material::transmission)
        .def_rw("emission", &Material::emission)
        .def_rw("emission_strength", &Material::emissionStrength)
        .def_rw("opacity", &Material::opacity);
}
