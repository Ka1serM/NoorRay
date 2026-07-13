#include "Bindings.h"

#include "Raytracing/Sellmeier.h"

namespace nb = nanobind;

void bindSellmeierCoefficients(nb::module_& module)
{
    nb::class_<SellmeierCoefficients>(module, "SellmeierCoefficients")
        .def(nb::init<>())
        .def_rw("b", &SellmeierCoefficients::b)
        .def_rw("c", &SellmeierCoefficients::c);
}
