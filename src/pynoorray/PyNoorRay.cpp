#include <nanobind/nanobind.h>

#include "PyRenderSession.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_pynoorray, m)
{
    nb::class_<PyRenderSession>(m, "RenderSession")
        .def(nb::init<uint32_t, uint32_t>(), "width"_a, "height"_a)
        .def("load_scene", &PyRenderSession::loadScene, "path"_a)
        .def("set_gaussian_opacity", &PyRenderSession::setGaussianOpacity, "index"_a, "opacity"_a)
        .def("render", &PyRenderSession::render, "spp"_a = 1)
        .def_prop_ro("width", &PyRenderSession::width)
        .def_prop_ro("height", &PyRenderSession::height)
        .def_prop_ro("gaussian_count", &PyRenderSession::gaussianCount);
}
