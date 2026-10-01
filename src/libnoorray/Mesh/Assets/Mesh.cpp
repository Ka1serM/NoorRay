#include "Mesh.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include <glm/common.hpp>
#include <glm/geometric.hpp>

#include "Scene/Scene.h"

using glm::normalize;
using glm::vec2;
using glm::vec3;

namespace {

int16_t snorm16(const float value)
{
    return static_cast<int16_t>(std::lround(std::clamp(value, -1.0f, 1.0f) * 32767.0f));
}

float unsnorm16(const int16_t value)
{
    return std::max(static_cast<float>(value) / 32767.0f, -1.0f);
}

// One vertex of a generated shape, which has a single UV channel.
void addVertex(MeshStreams& streams, const vec3 position, const vec3 normal,
    const vec3 tangent, const vec2 uv)
{
    streams.positions.push_back(position);
    streams.tangents.emplace_back(tangent, normal, 1.0f);
    streams.uvs.push_back(uv);
}

}

TangentFrame::TangentFrame(const vec3 tangent, const vec3 normal, const float bitangentSign)
    : tangentX{snorm16(tangent.x), snorm16(tangent.y), snorm16(tangent.z), 0},
      tangentZ{snorm16(normal.x), snorm16(normal.y), snorm16(normal.z),
          bitangentSign < 0.0f ? int16_t{-32767} : int16_t{32767}}
{
}

vec3 TangentFrame::tangent() const
{
    return {unsnorm16(tangentX[0]), unsnorm16(tangentX[1]), unsnorm16(tangentX[2])};
}

vec3 TangentFrame::normal() const
{
    return {unsnorm16(tangentZ[0]), unsnorm16(tangentZ[1]), unsnorm16(tangentZ[2])};
}

float TangentFrame::bitangentSign() const
{
    return tangentZ[3] < 0 ? -1.0f : 1.0f;
}

std::vector<MeshSection> sortTrianglesBySlot(std::vector<uint32_t>& indices,
    const std::span<const uint32_t> triangleSlots)
{
    const std::size_t triangleCount = indices.size() / 3;
    if (triangleSlots.size() != triangleCount)
        throw std::invalid_argument(std::to_string(triangleSlots.size())
            + " material slots for " + std::to_string(triangleCount) + " triangles");
    std::vector<uint32_t> order(triangleCount);
    std::ranges::iota(order, 0u);
    std::ranges::stable_sort(order, {}, [&](const uint32_t triangle) {
        return triangleSlots[triangle];
    });
    std::vector<uint32_t> sorted(indices.size());
    std::vector<MeshSection> sections;
    for (uint32_t target = 0; target < triangleCount; ++target) {
        const uint32_t source = order[target];
        std::copy_n(indices.begin() + std::size_t{source} * 3, 3,
            sorted.begin() + std::size_t{target} * 3);
        if (sections.empty() || sections.back().slot != triangleSlots[source])
            sections.push_back({triangleSlots[source], target, 0u});
        ++sections.back().triangleCount;
    }
    indices = std::move(sorted);
    return sections;
}

std::vector<MeshSection> singleSection(const std::vector<uint32_t>& indices)
{
    return {{0u, 0u, static_cast<uint32_t>(indices.size() / 3)}};
}

MeshGeometry::MeshGeometry(MeshStreams streams)
    : uvCount(streams.uvCount), sections(std::move(streams.sections))
{
    if (!streams.positions.empty()) {
        boundsMin = boundsMax = streams.positions.front();
        for (const vec3& position : streams.positions) {
            boundsMin = glm::min(boundsMin, position);
            boundsMax = glm::max(boundsMax, position);
        }
    }
    const auto owned = std::make_shared<const MeshStreams>(std::move(streams));
    positions = owned->positions;
    tangents = owned->tangents;
    uvs = owned->uvs;
    colors = owned->colors;
    indices = owned->indices;
    owner = owned;
}

Mesh Mesh::CreateCube(Scene& scene, const std::string& name)
{
    MeshStreams streams;
    constexpr float h = 0.5f;
    const vec3 faceNormals[6] = {
        { 0,  0,  1}, { 0,  0, -1},
        { 1,  0,  0}, {-1,  0,  0},
        { 0,  1,  0}, { 0, -1,  0}
    };
    const vec3 tangents[6] = {
        {1, 0, 0}, {-1, 0, 0},
        {0, 0, -1}, {0, 0, 1},
        {1, 0, 0}, {1, 0, 0}
    };
    const vec3 bitangents[6] = {
        {0, 1, 0}, {0, 1, 0},
        {0, 1, 0}, {0, 1, 0},
        {0, 0, -1}, {0, 0, 1}
    };
    for (uint32_t face = 0; face < 6; ++face) {
        const vec3 normal = faceNormals[face];
        const vec3 tangent = tangents[face];
        const vec3 bitangent = bitangents[face];
        const vec3 corners[4] = {
            normal * h + (-tangent - bitangent) * h,
            normal * h + ( tangent - bitangent) * h,
            normal * h + ( tangent + bitangent) * h,
            normal * h + (-tangent + bitangent) * h
        };
        const vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        const uint32_t first = face * 4;
        for (int i = 0; i < 4; ++i)
            addVertex(streams, corners[i], normal, tangent, uvs[i]);
        streams.indices.insert(streams.indices.end(),
            {first, first + 1, first + 2, first, first + 2, first + 3});
    }
    streams.sections = singleSection(streams.indices);
    return Mesh(scene, name, MeshGeometry(std::move(streams)));
}

Mesh Mesh::CreatePlane(Scene& scene, const std::string& name)
{
    MeshStreams streams;
    constexpr float halfSize = 0.5f;
    const vec3 positions[4] = {
        {-halfSize, -halfSize, 0.0f},
        { halfSize, -halfSize, 0.0f},
        { halfSize,  halfSize, 0.0f},
        {-halfSize,  halfSize, 0.0f}
    };
    const vec2 uvs[4] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    for (int i = 0; i < 4; ++i)
        addVertex(streams, positions[i], {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, uvs[i]);
    streams.indices = {0, 1, 2, 2, 3, 0};
    streams.sections = singleSection(streams.indices);
    return Mesh(scene, name, MeshGeometry(std::move(streams)));
}

Mesh Mesh::CreateSphere(Scene& scene, const std::string& name, const uint32_t latSeg,
    const uint32_t lonSeg)
{
    if (latSeg < 2 || lonSeg < 3)
        throw std::runtime_error("Sphere segments too low. Use at least 2 latitude and 3 longitude segments.");

    MeshStreams streams;
    constexpr float radius = 0.5f;
    for (uint32_t lat = 0; lat <= latSeg; ++lat) {
        const float theta = std::numbers::pi_v<float> * lat / latSeg;
        for (uint32_t lon = 0; lon <= lonSeg; ++lon) {
            const float phi = 2.0f * std::numbers::pi_v<float> * lon / lonSeg;
            const vec3 normal = {std::cos(phi) * std::sin(theta),
                std::sin(phi) * std::sin(theta), std::cos(theta)};
            // At the poles every vertex of a row shares one position, where
            // the tangent along the latitude is undefined; each pole takes a
            // fixed one so its vertices agree.
            const vec3 tangent = lat == 0 ? vec3{1.0f, 0.0f, 0.0f}
                : lat == latSeg ? vec3{-1.0f, 0.0f, 0.0f}
                : normalize(vec3{-std::sin(phi), std::cos(phi), 0.0f});
            addVertex(streams, normal * radius, normal, tangent,
                {static_cast<float>(lon) / lonSeg, static_cast<float>(lat) / latSeg});
        }
    }
    for (uint32_t lat = 0; lat < latSeg; ++lat) {
        for (uint32_t lon = 0; lon < lonSeg; ++lon) {
            const uint32_t i0 = lat * (lonSeg + 1) + lon;
            const uint32_t i1 = (lat + 1) * (lonSeg + 1) + lon;
            const uint32_t i2 = i0 + 1;
            const uint32_t i3 = i1 + 1;
            streams.indices.insert(streams.indices.end(), {i0, i2, i1, i2, i3, i1});
        }
    }
    streams.sections = singleSection(streams.indices);
    return Mesh(scene, name, MeshGeometry(std::move(streams)));
}

Mesh Mesh::CreateDisk(Scene& scene, const std::string& name, const uint32_t segments)
{
    if (segments < 3)
        throw std::runtime_error("Disk requires at least 3 segments");

    MeshStreams streams;
    constexpr float radius = 0.5f;
    const vec3 normal = {0.0f, 0.0f, 1.0f};
    addVertex(streams, {0.0f, 0.0f, 0.0f}, normal, {1.0f, 0.0f, 0.0f}, {0.5f, 0.5f});
    for (uint32_t i = 0; i <= segments; ++i) {
        const float angle = static_cast<float>(i) / segments * 2.0f * std::numbers::pi_v<float>;
        const float x = std::cos(angle) * radius;
        const float y = std::sin(angle) * radius;
        addVertex(streams, {x, y, 0.0f}, normal, {-std::sin(angle), std::cos(angle), 0.0f},
            {0.5f + x, 0.5f + y});
    }
    for (uint32_t i = 1; i <= segments; ++i)
        streams.indices.insert(streams.indices.end(), {0, i, i + 1});
    streams.sections = singleSection(streams.indices);
    return Mesh(scene, name, MeshGeometry(std::move(streams)));
}

Mesh::Mesh(Scene& scene, std::string name, MeshGeometry value)
    : scene(scene), path(std::move(name)), geometry(std::move(value))
{
    validate();
}

Mesh::Mesh(Mesh&& other) noexcept
    : noorrhi::Shared<nr::graphics::Mesh>(std::move(other)),
      scene(other.scene), path(std::move(other.path)), index(other.index),
      gpuDirty(other.gpuDirty), geometry(std::move(other.geometry)),
      slotCount(other.slotCount),
      positionBuffer(std::move(other.positionBuffer)),
      tangentBuffer(std::move(other.tangentBuffer)),
      uvBuffer(std::move(other.uvBuffer)),
      colorBuffer(std::move(other.colorBuffer)),
      indexBuffer(std::move(other.indexBuffer)),
      sectionBuffer(std::move(other.sectionBuffer)),
      blases(std::move(other.blases))
{
}

void Mesh::replaceGeometry(MeshGeometry value)
{
    scene.synchronizeBeforeMutation();
    geometry = std::move(value);
    validate();
    gpuDirty = true;
    scene.markMeshChanged(*this);
    scene.setDirtyFlag(Accumulation);
}

void Mesh::validate()
{
    const std::size_t vertexCount = geometry.positions.size();
    if (geometry.tangents.size() != vertexCount || geometry.uvCount < 1u
        || geometry.uvs.size() != vertexCount * geometry.uvCount
        || (!geometry.colors.empty() && geometry.colors.size() != vertexCount))
        throw std::invalid_argument("Mesh " + path + " has streams of different vertex counts");
    uint32_t next = 0;
    slotCount = 0;
    for (const MeshSection& section : geometry.sections) {
        slotCount = std::max(slotCount, section.slot + 1);
        if (section.firstTriangle != next || section.triangleCount == 0)
            throw std::invalid_argument("Mesh " + path + " has a section at triangle "
                + std::to_string(section.firstTriangle) + " that does not continue at "
                + std::to_string(next) + " or is empty");
        next += section.triangleCount;
    }
    if (next != geometry.indices.size() / 3)
        throw std::invalid_argument("Mesh " + path + " has sections covering "
            + std::to_string(next) + " of " + std::to_string(geometry.indices.size() / 3) + " triangles");
}

namespace {
template<class T>
void uploadStream(noorrhi::Device& device, noorrhi::Buffer<T>& buffer, const std::span<const T> data)
{
    if (data.empty()) {
        buffer = {};
        return;
    }
    if (!buffer || buffer.size() != data.size())
        buffer = device.buffer<T>(data.size());
    buffer.upload(data);
}

std::uint64_t address(const auto& buffer)
{
    return buffer ? buffer.ptr().address : 0u;
}
}

void Mesh::upload(noorrhi::Device& device)
{
    if (!gpuDirty || geometry.positions.empty() || geometry.indices.empty())
        return;
    uploadStream(device, positionBuffer, geometry.positions);
    uploadStream(device, tangentBuffer, geometry.tangents);
    uploadStream(device, uvBuffer, geometry.uvs);
    uploadStream(device, colorBuffer, geometry.colors);
    uploadStream(device, indexBuffer, geometry.indices);
    std::vector<nr::graphics::SectionRecord> records;
    records.reserve(geometry.sections.size());
    for (const MeshSection& section : geometry.sections)
        records.push_back({section.firstTriangle, section.slot, section.castsShadow ? 1u : 0u});
    uploadStream(device, sectionBuffer, std::span<const nr::graphics::SectionRecord>(records));
    blases.clear();
    gpuDirty = false;

    if (!*this)
        allocate(device);
    data = nr::graphics::Mesh{
        address(positionBuffer),
        address(tangentBuffer),
        address(uvBuffer),
        address(colorBuffer),
        address(indexBuffer),
        address(sectionBuffer),
        static_cast<uint32_t>(geometry.positions.size()),
        static_cast<uint32_t>(geometry.indices.size()),
        static_cast<uint32_t>(geometry.sections.size()),
        geometry.uvCount,
        {geometry.boundsMin.x, geometry.boundsMin.y, geometry.boundsMin.z},
        {geometry.boundsMax.x, geometry.boundsMax.y, geometry.boundsMax.z},
    };
    commit();
}

const noorrhi::AccelerationStructure& Mesh::blas(noorrhi::Device& device, const std::vector<bool>& opacity)
{
    for (const auto& [key, structure] : blases)
        if (key == opacity)
            return structure;
    std::vector<noorrhi::TriangleGeometry> geometries;
    geometries.reserve(geometry.sections.size());
    for (std::size_t i = 0; i < geometry.sections.size(); ++i)
        geometries.push_back({
            noorrhi::GpuPtr<noorrhi::float3>{positionBuffer.ptr().address},
            noorrhi::GpuPtr<std::uint32_t>{indexBuffer.ptr().address
                + std::uint64_t{geometry.sections[i].firstTriangle} * 3u * sizeof(std::uint32_t)},
            geometry.sections[i].triangleCount, sizeof(glm::vec3), opacity[i]});
    return blases.emplace_back(opacity, device.build_blas(geometries)).second;
}

void Mesh::releaseGpu()
{
    gpuDirty = true;
    release();
    blases.clear();
    positionBuffer = {};
    tangentBuffer = {};
    uvBuffer = {};
    colorBuffer = {};
    indexBuffer = {};
    sectionBuffer = {};
}
