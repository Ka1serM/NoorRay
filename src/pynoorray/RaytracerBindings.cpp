#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "Backend/OptiX/Runtime/Raytracer.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindRaytracer(nb::module_& module)
{
    nb::class_<Raytracer>(module, "Raytracer")
        .def("render_frame", [](Raytracer& raytracer) {
            raytracer.renderFrame();
        })
        .def("resize", &Raytracer::resize, "width"_a, "height"_a)
        .def("set_aov_enabled", &Raytracer::setAovEnabled, "enabled"_a)
        .def("set_stats_enabled", &Raytracer::setStatsEnabled, "enabled"_a)
        .def("set_timing_enabled", &Raytracer::setTimingEnabled, "enabled"_a)
        .def("print_kernel_stats", &Raytracer::printKernelStats)
        .def("debug_save", &Raytracer::debugSave, "path"_a)
        .def_prop_ro("width", &Raytracer::getWidth)
        .def_prop_ro("height", &Raytracer::getHeight)
        .def_prop_ro("beauty", nb::overload_cast<>(&Raytracer::getOutputColor),
            nb::rv_policy::reference_internal)
        .def_prop_ro("albedo", nb::overload_cast<>(&Raytracer::getOutputAlbedo),
            nb::rv_policy::reference_internal)
        .def_prop_ro("normal", nb::overload_cast<>(&Raytracer::getOutputNormal),
            nb::rv_policy::reference_internal)
        .def_prop_ro("cryptomatte", nb::overload_cast<>(&Raytracer::getOutputCrypto),
            nb::rv_policy::reference_internal)
        .def_prop_ro("position", nb::overload_cast<>(&Raytracer::getOutputPosition),
            nb::rv_policy::reference_internal)
        .def_prop_ro("scratch_capacity", &Raytracer::getScratchCapacity);
}
