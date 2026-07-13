#include "Bindings.h"

#include "Vulkan/Context.h"

namespace nb = nanobind;

void bindContext(nb::module_& module)
{
    nb::class_<Context>(module, "Context");
}
