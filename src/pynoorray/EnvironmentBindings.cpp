#include "Bindings.h"

#include "Scene/Resources/Environment.h"
#include "Scene/Resources/Texture.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindEnvironment(nb::module_& module)
{
    nb::class_<Environment>(module, "Environment")
        .def("set_hdri_texture", &Environment::setHdriTexture, "texture"_a)
        .def("clear_hdri_texture", &Environment::clearHdriTexture)
        .def_prop_rw("color", [](const Environment& value) { return value.color; },
            [](Environment& value, const glm::vec3& color) {
                value.color = color;
                value.updateDerivedSettings();
            })
        .def_prop_rw("rotation", [](const Environment& value) { return value.rotation; },
            [](Environment& value, const float rotation) {
                value.rotation = rotation;
                value.updateDerivedSettings();
            })
        .def_prop_rw("visible_exposure", [](const Environment& value) { return value.visibleExposure; },
            [](Environment& value, const float exposure) {
                value.visibleExposure = exposure;
                value.updateDerivedSettings();
            })
        .def_prop_rw("lighting_exposure", [](const Environment& value) { return value.lightingExposure; },
            [](Environment& value, const float exposure) {
                value.lightingExposure = exposure;
                value.updateDerivedSettings();
            });
}
