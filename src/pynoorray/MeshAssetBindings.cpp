#include "Bindings.h"

#include <nanobind/stl/string.h>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>

#include "Materials/MaterialX/MaterialXDocument.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Scene/Scene.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace
{

MaterialX::DocumentPtr materialDocument(const std::string& xml)
{
    if (xml.empty())
        return nr::materialx::defaultMaterial();
    MaterialX::DocumentPtr document = MaterialX::createDocument();
    MaterialX::readFromXmlString(document, xml);
    return document;
}

}

void bindMeshAsset(nb::module_& module)
{
    nb::class_<MeshAsset>(module, "MeshAsset")
        .def_static("create_cube",
            [](Scene& scene, const std::string& name, const std::string& materialXml) {
                return MeshAsset::CreateCube(scene, name, materialDocument(materialXml));
            },
            "scene"_a, "name"_a = "Cube", "material_xml"_a = "")
        .def_static("create_plane",
            [](Scene& scene, const std::string& name, const std::string& materialXml) {
                return MeshAsset::CreatePlane(scene, name, materialDocument(materialXml));
            },
            "scene"_a, "name"_a = "Plane", "material_xml"_a = "")
        .def_static("create_sphere",
            [](Scene& scene, const std::string& name, const std::string& materialXml,
                const uint32_t latitudeSegments, const uint32_t longitudeSegments) {
                return MeshAsset::CreateSphere(scene, name, materialDocument(materialXml),
                    latitudeSegments, longitudeSegments);
            },
            "scene"_a, "name"_a = "Sphere", "material_xml"_a = "",
            "latitude_segments"_a = 64, "longitude_segments"_a = 64)
        .def_static("create_disk",
            [](Scene& scene, const std::string& name, const std::string& materialXml,
                const uint32_t segments) {
                return MeshAsset::CreateDisk(scene, name, materialDocument(materialXml),
                    segments);
            },
            "scene"_a, "name"_a = "Disk", "material_xml"_a = "", "segments"_a = 64)
        .def_prop_ro("name", &MeshAsset::getName)
        .def_prop_ro("mesh_index", &MeshAsset::getMeshIndex);
}
