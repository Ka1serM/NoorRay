#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "Scene/MeshInstance.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindMeshInstance(nb::module_& module)
{
    nb::class_<MeshInstance, SceneObject>(module, "MeshInstance")
        .def(nb::init<Scene&, const std::string&, const MeshAssetRef&, const Transform&>(),
            "scene"_a, "name"_a, "mesh_asset"_a, "transform"_a = Transform{},
            nb::keep_alive<1, 2>())
        .def_prop_ro("mesh_asset", &MeshInstance::getMeshAssetRef)
        .def_prop_ro("mesh_index", &MeshInstance::getMeshIndex);
}
