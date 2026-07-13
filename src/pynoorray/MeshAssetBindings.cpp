#include "Bindings.h"

#include <nanobind/stl/string.h>

#include "Mesh/MeshAsset.h"
#include "Scene/Scene.h"

namespace nb = nanobind;
using namespace nb::literals;

void bindMeshAsset(nb::module_& module)
{
    nb::class_<MeshAsset>(module, "MeshAsset")
        .def_static("create_cube", &MeshAsset::CreateCube,
            "scene"_a, "name"_a = "Cube", "material"_a = Material{})
        .def_static("create_plane", &MeshAsset::CreatePlane,
            "scene"_a, "name"_a = "Plane", "material"_a = Material{})
        .def_static("create_sphere", &MeshAsset::CreateSphere,
            "scene"_a, "name"_a = "Sphere", "material"_a = Material{},
            "latitude_segments"_a = 64, "longitude_segments"_a = 64)
        .def_static("create_disk", &MeshAsset::CreateDisk,
            "scene"_a, "name"_a = "Disk", "material"_a = Material{}, "segments"_a = 64)
        .def_prop_ro("name", &MeshAsset::getName)
        .def_prop_ro("mesh_index", &MeshAsset::getMeshIndex);
}
