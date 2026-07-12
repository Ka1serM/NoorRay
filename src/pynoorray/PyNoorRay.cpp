#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "PyRenderSession.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_pynoorray, module)
{
    nb::class_<PyRenderSession>(module, "RenderSession")
        .def(nb::init<uint32_t, uint32_t>(), "width"_a, "height"_a)
        .def("import_file", &PyRenderSession::importFile, "path"_a)
        .def("read_scene", &PyRenderSession::readScene, "path"_a)
        .def("load_scene", &PyRenderSession::loadScene, "path"_a)
        .def("add_perspective_camera", &PyRenderSession::addPerspectiveCamera,
            "position"_a, "focal_length_mm"_a = 50.0f)
        .def("set_camera_to_world", &PyRenderSession::setCameraToWorld, "matrix"_a)
        .def("set_camera_focal_length", &PyRenderSession::setCameraFocalLength, "focal_length_mm"_a)
        .def("set_samples", &PyRenderSession::setSamples, "samples"_a)
        .def("set_max_samples", &PyRenderSession::setMaxSamples, "samples"_a)
        .def("set_max_bounces", &PyRenderSession::setMaxBounces, "bounces"_a)
        .def("set_exposure", &PyRenderSession::setExposure, "exposure"_a)
        .def("render", &PyRenderSession::render, "spp"_a = 1)
        .def_prop_ro("width", &PyRenderSession::width)
        .def_prop_ro("height", &PyRenderSession::height)
        .def_prop_ro("gaussian_count", &PyRenderSession::gaussianCount);
}
