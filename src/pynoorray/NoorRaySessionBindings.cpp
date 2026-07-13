#include "Bindings.h"

#include "NoorRaySession.h"
#include "Raytracing/Raytracer.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindNoorRaySession(nb::module_& module)
{
    nb::class_<noorray::NoorRaySession>(module, "NoorRaySession")
        .def(nb::init<>())
        .def_prop_ro("context", [](noorray::NoorRaySession& session) -> Context& {
            return session.context;
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("scene", [](noorray::NoorRaySession& session) -> Scene& {
            return session.scene;
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("raytracer", [](noorray::NoorRaySession& session) -> Raytracer& {
            return *session.raytracer;
        }, nb::rv_policy::reference_internal)
        .def_ro("headless", &noorray::NoorRaySession::headless);
}
