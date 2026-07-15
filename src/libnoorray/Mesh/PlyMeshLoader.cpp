#include "PlyMeshLoader.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include <tinyply.h>
#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

namespace {
using tinyply::PlyData;
using tinyply::Type;

template<typename T>
T readValue(const PlyData& data, const size_t index)
{
    T value{};
    std::memcpy(&value, data.buffer.get_const() + index * sizeof(T), sizeof(T));
    return value;
}

double numberAt(const PlyData& data, const size_t index)
{
    switch (data.t) {
    case Type::INT8: return readValue<int8_t>(data, index);
    case Type::UINT8: return readValue<uint8_t>(data, index);
    case Type::INT16: return readValue<int16_t>(data, index);
    case Type::UINT16: return readValue<uint16_t>(data, index);
    case Type::INT32: return readValue<int32_t>(data, index);
    case Type::UINT32: return readValue<uint32_t>(data, index);
    case Type::FLOAT32: return readValue<float>(data, index);
    case Type::FLOAT64: return readValue<double>(data, index);
    case Type::INVALID: break;
    }
    throw std::runtime_error("PLY property has an invalid numeric type");
}

bool hasProperties(const tinyply::PlyElement& element, const std::initializer_list<const char*> names)
{
    for (const char* name : names) {
        bool found = false;
        for (const auto& property : element.properties)
            if (property.name == name) {
                found = true;
                break;
            }
        if (!found) return false;
    }
    return true;
}

const tinyply::PlyElement* findElement(
    const std::vector<tinyply::PlyElement>& elements, const std::string& name)
{
    for (const auto& element : elements)
        if (element.name == name) return &element;
    return nullptr;
}

std::ifstream openPly(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Failed to open PLY file: " + path.string());
    return stream;
}

void buildNormalsAndTangents(PlyMeshData& mesh, const bool hasNormals)
{
    if (!hasNormals)
        for (Vertex& vertex : mesh.vertices) vertex.normal = glm::vec3(0.f);

    if (!hasNormals) {
        for (size_t i = 0; i < mesh.indices.size(); i += 3) {
            Vertex& a = mesh.vertices[mesh.indices[i]];
            Vertex& b = mesh.vertices[mesh.indices[i + 1]];
            Vertex& c = mesh.vertices[mesh.indices[i + 2]];
            const glm::vec3 normal = glm::cross(b.position - a.position, c.position - a.position);
            a.normal += normal;
            b.normal += normal;
            c.normal += normal;
        }
    }

    for (Vertex& vertex : mesh.vertices) {
        vertex.normal = glm::dot(vertex.normal, vertex.normal) > 0.f
            ? glm::normalize(vertex.normal) : glm::vec3(0.f, 1.f, 0.f);
        const glm::vec3 helper = std::abs(vertex.normal.y) < .99f
            ? glm::vec3(0.f, 1.f, 0.f) : glm::vec3(1.f, 0.f, 0.f);
        vertex.tangent = glm::normalize(glm::cross(helper, vertex.normal));
        vertex.tangentSign = 1.f;
    }
}
}

bool PlyMeshLoader::HasFaces(const std::filesystem::path& path)
{
    try {
        auto stream = openPly(path);
        tinyply::PlyFile file;
        if (!file.parse_header(stream)) return false;
        const auto elements = file.get_elements();
        const tinyply::PlyElement* face = findElement(elements, "face");
        return face && face->size > 0
            && (hasProperties(*face, {"vertex_indices"})
                || hasProperties(*face, {"vertex_index"}));
    } catch (...) {
        return false;
    }
}

PlyMeshData PlyMeshLoader::Load(const std::filesystem::path& path)
{
    auto stream = openPly(path);
    tinyply::PlyFile file;
    if (!file.parse_header(stream))
        throw std::runtime_error("Failed to parse PLY header: " + path.string());

    const auto elements = file.get_elements();
    const tinyply::PlyElement* vertexElement = findElement(elements, "vertex");
    const tinyply::PlyElement* faceElement = findElement(elements, "face");
    if (!vertexElement || !hasProperties(*vertexElement, {"x", "y", "z"}))
        throw std::runtime_error("PLY mesh has no x/y/z vertex positions: " + path.string());
    if (!faceElement || faceElement->size == 0)
        throw std::runtime_error("PLY mesh has no faces: " + path.string());

    const bool hasNormals = hasProperties(*vertexElement, {"nx", "ny", "nz"});
    std::vector<std::string> uvNames;
    if (hasProperties(*vertexElement, {"u", "v"})) uvNames = {"u", "v"};
    else if (hasProperties(*vertexElement, {"s", "t"})) uvNames = {"s", "t"};

    const std::string faceProperty = hasProperties(*faceElement, {"vertex_indices"})
        ? "vertex_indices" : hasProperties(*faceElement, {"vertex_index"})
        ? "vertex_index" : std::string{};
    if (faceProperty.empty())
        throw std::runtime_error("PLY faces have no vertex index list: " + path.string());

    const std::shared_ptr<PlyData> positions =
        file.request_properties_from_element("vertex", {"x", "y", "z"});
    const std::shared_ptr<PlyData> normals = hasNormals
        ? file.request_properties_from_element("vertex", {"nx", "ny", "nz"}) : nullptr;
    const std::shared_ptr<PlyData> uvs = !uvNames.empty()
        ? file.request_properties_from_element("vertex", uvNames) : nullptr;
    const std::shared_ptr<PlyData> faces =
        file.request_properties_from_element("face", {faceProperty}, 0);
    file.read(stream);

    PlyMeshData mesh;
    mesh.vertices.resize(positions->count);
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        Vertex& vertex = mesh.vertices[i];
        vertex.position = glm::vec3(
            numberAt(*positions, i * 3), numberAt(*positions, i * 3 + 1), numberAt(*positions, i * 3 + 2));
        if (normals)
            vertex.normal = glm::vec3(
                numberAt(*normals, i * 3), numberAt(*normals, i * 3 + 1), numberAt(*normals, i * 3 + 2));
        if (uvs)
            vertex.uv = glm::vec2(numberAt(*uvs, i * 2), 1.0 - numberAt(*uvs, i * 2 + 1));
    }

    const size_t scalarSize = tinyply::PropertyTable.at(faces->t).stride;
    const size_t totalFaceIndices = faces->buffer.size_bytes() / scalarSize;
    const size_t fixedVertexCount = faces->count > 0 ? totalFaceIndices / faces->count : 0;
    auto checkedIndex = [&](const size_t offset) {
        const double value = numberAt(*faces, offset);
        if (value < 0.0 || value >= static_cast<double>(mesh.vertices.size()))
            throw std::runtime_error("PLY face index is out of bounds: " + path.string());
        return static_cast<uint32_t>(value);
    };

    size_t sourceOffset = 0;
    for (size_t face = 0; face < faces->count; ++face) {
        const size_t vertexCount = faces->list_sizes.empty() ? fixedVertexCount : faces->list_sizes[face];
        if (vertexCount < 3) {
            sourceOffset += vertexCount;
            continue;
        }
        const uint32_t first = checkedIndex(sourceOffset);
        for (size_t corner = 1; corner + 1 < vertexCount; ++corner) {
            const uint32_t second = checkedIndex(sourceOffset + corner);
            const uint32_t third = checkedIndex(sourceOffset + corner + 1);
            mesh.indices.insert(mesh.indices.end(), {first, second, third});
        }
        sourceOffset += vertexCount;
    }
    if (mesh.indices.empty())
        throw std::runtime_error("PLY mesh contains no triangles: " + path.string());

    buildNormalsAndTangents(mesh, hasNormals);
    return mesh;
}
