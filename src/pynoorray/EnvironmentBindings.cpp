#include "Bindings.h"

#include "Environment/Environment.h"
#include "Texture/Texture.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindEnvironment(nb::module_& module)
{
    nb::class_<Environment>(module, "Environment")
        .def("set_hdri_texture", &Environment::setHdriTexture, "texture"_a)
        .def("clear_hdri_texture", &Environment::clearHdriTexture)
        // The authored color is stored directly in the shared shader record.
        .def_prop_rw("color", &Environment::getColor, &Environment::setColor)
        .def_prop_rw("rotation", &Environment::getRotation, &Environment::setRotation)
        .def_prop_rw("visible_exposure", &Environment::getVisibleExposure,
            &Environment::setVisibleExposure)
        .def_prop_rw("lighting_exposure", &Environment::getLightingExposure,
            &Environment::setLightingExposure);
}
