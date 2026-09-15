#include "Bindings.h"

#include <nanobind/stl/string.h>

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>

#include "Materials/MaterialX/MaterialXDocument.h"
#include "Mesh/Assets/Mesh.h"
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

void bindMesh(nb::module_& module)
{
    nb::class_<Mesh>(module, "Mesh")
        .def_static("create_cube",
            [](Scene& scene, const std::string& name, const std::string& materialXml) {
                return Mesh::CreateCube(scene, name, materialDocument(materialXml));
            },
            "scene"_a, "name"_a = "Cube", "material_xml"_a = "")
        .def_static("create_plane",
            [](Scene& scene, const std::string& name, const std::string& materialXml) {
                return Mesh::CreatePlane(scene, name, materialDocument(materialXml));
            },
            "scene"_a, "name"_a = "Plane", "material_xml"_a = "")
        .def_static("create_sphere",
            [](Scene& scene, const std::string& name, const std::string& materialXml,
                const uint32_t latitudeSegments, const uint32_t longitudeSegments) {
                return Mesh::CreateSphere(scene, name, materialDocument(materialXml),
                    latitudeSegments, longitudeSegments);
            },
            "scene"_a, "name"_a = "Sphere", "material_xml"_a = "",
            "latitude_segments"_a = 64, "longitude_segments"_a = 64)
        .def_static("create_disk",
            [](Scene& scene, const std::string& name, const std::string& materialXml,
                const uint32_t segments) {
                return Mesh::CreateDisk(scene, name, materialDocument(materialXml),
                    segments);
            },
            "scene"_a, "name"_a = "Disk", "material_xml"_a = "", "segments"_a = 64)
        .def_prop_ro("name", &Mesh::getName)
        .def_prop_ro("mesh_index", &Mesh::getMeshIndex);
}
