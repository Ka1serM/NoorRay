#include "NoorRay.Python/Blender/StrandReader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <utility>

namespace NoorRay::Python {

namespace {

//! Below this many strand points the tubes have more sides.
constexpr std::size_t fewPoints = 400000;
constexpr std::size_t curveSteps = 2;

struct Strands {
    //! The first point of each strand, then the count of all points.
    std::vector<std::uint32_t> starts;
    std::vector<Core::float3> positions;
    std::vector<float> radii;
    //! One per strand, or empty.
    std::vector<Core::float2> uvs;
};

Core::float3 perpendicular(const Core::float3 direction) {
    const Core::float3 axis = std::abs(direction.x) < 0.8f ? Core::float3{1.0f, 0.0f, 0.0f} : Core::float3{0.0f, 1.0f, 0.0f};
    return Core::normalize(Core::cross(direction, axis));
}

Scene::Mesh makeTubes(const Strands& strands, std::string name, const Scene::MaterialHandle material) {
    Scene::Mesh mesh;
    mesh.name = std::move(name);
    const std::size_t sides = strands.positions.size() < fewPoints ? 5 : 3;
    const std::size_t strandCount = strands.starts.size() - 1;
    mesh.positions.reserve(strands.positions.size() * sides);
    mesh.normals.reserve(strands.positions.size() * sides);
    mesh.uvs[0].reserve(strands.positions.size() * sides);
    for (std::size_t strand = 0; strand < strandCount; ++strand) {
        const std::size_t first = strands.starts[strand];
        const std::size_t count = strands.starts[strand + 1] - first;
        if (count < 2)
            continue;
        Core::float3 side{};
        const Core::float2 uv = strands.uvs.empty() ? Core::float2{} : strands.uvs[strand];
        const auto base = static_cast<std::uint32_t>(mesh.positions.size());
        for (std::size_t point = 0; point < count; ++point) {
            const Core::float3 position = strands.positions[first + point];
            const Core::float3 before = strands.positions[first + (point > 0 ? point - 1 : 0)];
            const Core::float3 after = strands.positions[first + std::min(point + 1, count - 1)];
            const Core::float3 along = after - before;
            const Core::float3 tangent = Core::length(along) > 1e-12f ? Core::normalize(along) : Core::float3{0.0f, 0.0f, 1.0f};
            side = side - tangent * Core::dot(side, tangent);
            side = Core::length(side) > 1e-4f ? Core::normalize(side) : perpendicular(tangent);
            const Core::float3 up = Core::cross(tangent, side);
            for (std::size_t corner = 0; corner < sides; ++corner) {
                const float angle = 2.0f * Core::pi * float(corner) / float(sides);
                const Core::float3 normal = side * std::cos(angle) + up * std::sin(angle);
                mesh.positions.push_back(position + normal * strands.radii[first + point]);
                mesh.normals.push_back(Core::packOctahedral(normal));
                mesh.uvs[0].push_back(Core::packHalf2(uv.x, uv.y));
            }
        }
        for (std::size_t point = 0; point + 1 < count; ++point)
            for (std::size_t corner = 0; corner < sides; ++corner) {
                const auto next = static_cast<std::uint32_t>((corner + 1) % sides);
                const auto low = base + static_cast<std::uint32_t>(point * sides);
                const auto high = low + static_cast<std::uint32_t>(sides);
                const auto here = static_cast<std::uint32_t>(corner);
                mesh.indices.insert(mesh.indices.end(), {low + here, low + next, high + next, low + here, high + next, high + here});
            }
    }
    mesh.subsets.push_back({.firstIndex = 0, .indexCount = static_cast<std::uint32_t>(mesh.indices.size()), .material = material});
    mesh.computeBounds();
    return mesh;
}

Core::float3 catmullRom(const Core::float3 a, const Core::float3 b, const Core::float3 c, const Core::float3 d, const float t) {
    return (b * 2.0f + (c - a) * t + (a * 2.0f - b * 5.0f + c * 4.0f - d) * (t * t) + (b * 3.0f - a - c * 3.0f + d) * (t * t * t)) * 0.5f;
}

//! Cycles draws hair as Catmull-Rom curves through the points.
Strands smooth(const Strands& source) {
    Strands result;
    result.uvs = source.uvs;
    result.starts.push_back(0);
    for (std::size_t strand = 0; strand + 1 < source.starts.size(); ++strand) {
        const std::ptrdiff_t first = source.starts[strand];
        const std::ptrdiff_t last = std::ptrdiff_t(source.starts[strand + 1]) - 1;
        const auto at = [&](const std::ptrdiff_t index) { return source.positions[std::size_t(std::clamp(index, first, last))]; };
        for (std::ptrdiff_t point = first; point <= last; ++point)
            for (std::size_t step = 0; step < (point < last ? curveSteps : 1); ++step) {
                const float t = float(step) / float(curveSteps);
                const float next = source.radii[std::size_t(std::min(point + 1, last))];
                result.positions.push_back(catmullRom(at(point - 1), at(point), at(point + 1), at(point + 2), t));
                result.radii.push_back(source.radii[std::size_t(point)] * (1.0f - t) + next * t);
            }
        result.starts.push_back(static_cast<std::uint32_t>(result.positions.size()));
    }
    return result;
}

struct Sphere {
    std::vector<Core::float3> points;
    std::vector<std::uint32_t> indices;
};

Sphere makeSphere(const std::size_t subdivisions) {
    const float golden = 0.5f * (1.0f + std::sqrt(5.0f));
    Sphere sphere;
    for (const float sign : {-1.0f, 1.0f})
        for (const float other : {-golden, golden}) {
            sphere.points.push_back(Core::normalize(Core::float3{0.0f, sign, other}));
            sphere.points.push_back(Core::normalize(Core::float3{sign, other, 0.0f}));
            sphere.points.push_back(Core::normalize(Core::float3{other, 0.0f, sign}));
        }
    //! The faces of the icosahedron are the triples of neighbours, turned to face outwards.
    const float edge = 1.1f * Core::length(sphere.points[0] - sphere.points[2]);
    float shortest = edge * 10.0f;
    for (std::size_t other = 1; other < sphere.points.size(); ++other)
        shortest = std::min(shortest, Core::length(sphere.points[0] - sphere.points[other]));
    const auto near = [&](const std::size_t a, const std::size_t b) { return Core::length(sphere.points[a] - sphere.points[b]) < shortest * 1.1f; };
    for (std::uint32_t a = 0; a < 12; ++a)
        for (std::uint32_t b = a + 1; b < 12; ++b)
            for (std::uint32_t c = b + 1; c < 12; ++c) {
                if (!near(a, b) || !near(b, c) || !near(a, c))
                    continue;
                const Core::float3 normal = Core::cross(sphere.points[b] - sphere.points[a], sphere.points[c] - sphere.points[a]);
                const bool outward = Core::dot(normal, sphere.points[a]) > 0.0f;
                sphere.indices.insert(sphere.indices.end(), {a, outward ? b : c, outward ? c : b});
            }
    for (std::size_t level = 0; level < subdivisions; ++level) {
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::uint32_t> middles;
        const auto middle = [&](const std::uint32_t a, const std::uint32_t b) {
            const auto [found, inserted] = middles.try_emplace(std::minmax(a, b), static_cast<std::uint32_t>(sphere.points.size()));
            if (inserted)
                sphere.points.push_back(Core::normalize(sphere.points[a] + sphere.points[b]));
            return found->second;
        };
        std::vector<std::uint32_t> finer;
        for (std::size_t index = 0; index < sphere.indices.size(); index += 3) {
            const std::uint32_t a = sphere.indices[index];
            const std::uint32_t b = sphere.indices[index + 1];
            const std::uint32_t c = sphere.indices[index + 2];
            const std::uint32_t ab = middle(a, b);
            const std::uint32_t bc = middle(b, c);
            const std::uint32_t ca = middle(c, a);
            finer.insert(finer.end(), {a, ab, ca, ab, b, bc, ca, bc, c, ab, bc, ca});
        }
        sphere.indices = std::move(finer);
    }
    return sphere;
}

}

std::vector<float> StrandReader::attribute(const Object& data, const char* name, const char* field, const std::size_t components, const std::size_t count) {
    const Object found = reader.findItem(reader.get(data, "attributes"), name);
    if (!found)
        return {};
    const Object values = reader.get(found, "data");
    if (reader.size(values) != count)
        return {};
    std::vector<float> result(count * components);
    reader.foreachGet(values, field, std::span(result));
    return result;
}

Scene::Mesh StrandReader::readCurves(const Object& curves, const Scene::MaterialHandle material) {
    const std::size_t pointCount = reader.size(reader.get(curves, "points"));
    const std::size_t curveCount = reader.size(reader.get(curves, "curves"));
    std::vector<float> positions = attribute(curves, "position", "vector", 3, pointCount);
    std::vector<float> radii = attribute(curves, "radius", "value", 1, pointCount);
    const std::vector<float> uvs = attribute(curves, "surface_uv_coordinate", "vector", 2, curveCount);
    if (reader.failed() || positions.empty() || curveCount == 0)
        return {};
    std::vector<std::int32_t> firsts(curveCount + 1);
    reader.foreachGet(reader.get(curves, "curve_offset_data"), "value", std::span(firsts));
    if (reader.failed())
        return {};
    Strands strands;
    strands.starts.assign(firsts.begin(), firsts.end());
    strands.radii = std::move(radii);
    //! The radius of Cycles for curves without one.
    strands.radii.resize(pointCount, 0.005f);
    strands.positions.resize(pointCount);
    std::memcpy(strands.positions.data(), positions.data(), positions.size() * sizeof(float));
    for (std::size_t curve = 0; curve < uvs.size() / 2; ++curve)
        strands.uvs.push_back({uvs[curve * 2], uvs[curve * 2 + 1]});
    return makeTubes(smooth(strands), reader.text(curves, "name"), material);
}

Scene::Mesh StrandReader::readPoints(const Object& points, const Scene::MaterialHandle material) {
    const std::size_t count = reader.size(reader.get(points, "points"));
    const std::vector<float> positions = attribute(points, "position", "vector", 3, count);
    std::vector<float> radii = attribute(points, "radius", "value", 1, count);
    if (reader.failed() || positions.empty())
        return {};
    radii.resize(count, 0.01f);
    const Sphere sphere = makeSphere(count <= 2000 ? 2 : count <= 60000 ? 1 : 0);
    Scene::Mesh mesh;
    mesh.name = reader.text(points, "name");
    mesh.positions.reserve(count * sphere.points.size());
    mesh.normals.reserve(count * sphere.points.size());
    mesh.indices.reserve(count * sphere.indices.size());
    for (std::size_t point = 0; point < count; ++point) {
        const auto base = static_cast<std::uint32_t>(mesh.positions.size());
        const Core::float3 centre{positions[point * 3], positions[point * 3 + 1], positions[point * 3 + 2]};
        for (const Core::float3 direction : sphere.points) {
            mesh.positions.push_back(centre + direction * radii[point]);
            mesh.normals.push_back(Core::packOctahedral(direction));
        }
        for (const std::uint32_t index : sphere.indices)
            mesh.indices.push_back(base + index);
    }
    mesh.subsets.push_back({.firstIndex = 0, .indexCount = static_cast<std::uint32_t>(mesh.indices.size()), .material = material});
    mesh.computeBounds();
    return mesh;
}

std::vector<Scene::Mesh> StrandReader::readParticles(const Object& object, const bool render, const std::span<const Scene::MaterialHandle> slotMaterials) {
    std::vector<Scene::Mesh> meshes;
    const Object systems = reader.helper("particle_hair", std::array{object, Object::borrow(render ? Py_True : Py_False)});
    if (reader.failed())
        return meshes;
    for (const Object& system : reader.items(systems)) {
        const std::vector<Object> parts = reader.items(system);
        if (parts.size() != 4)
            return meshes;
        const auto slot = static_cast<std::size_t>(PyLong_AsLong(parts[0].get()));
        const auto steps = static_cast<std::size_t>(PyLong_AsLong(parts[1].get()));
        const std::span<const std::byte> positions = reader.bytes(parts[2]);
        const std::span<const std::byte> radii = reader.bytes(parts[3]);
        if (reader.failed() || radii.size() != steps * sizeof(float))
            return meshes;
        Strands strands;
        strands.positions.resize(positions.size() / sizeof(Core::float3));
        std::memcpy(strands.positions.data(), positions.data(), strands.positions.size() * sizeof(Core::float3));
        std::vector<float> profile(steps);
        std::memcpy(profile.data(), radii.data(), radii.size());
        for (std::size_t point = 0; point < strands.positions.size(); ++point) {
            if (point % steps == 0)
                strands.starts.push_back(static_cast<std::uint32_t>(point));
            strands.radii.push_back(profile[point % steps]);
        }
        strands.starts.push_back(static_cast<std::uint32_t>(strands.positions.size()));
        const Scene::MaterialHandle material = slotMaterials.empty() ? Scene::MaterialHandle() : slotMaterials[std::min(slot, slotMaterials.size() - 1)];
        meshes.push_back(makeTubes(strands, reader.text(object, "name") + " Hair", material));
    }
    return meshes;
}

std::vector<std::uint16_t> fillMask(const std::span<const Core::float3> positions, const std::span<const std::uint32_t> indices, const Core::Bounds& bounds,
                                    const Core::uint3 resolution) {
    constexpr std::size_t subsamples = 2;
    const std::size_t columns = resolution.x * subsamples;
    const std::size_t rows = resolution.y * subsamples;
    const Core::float3 extent = bounds.max - bounds.min;
    const Core::float3 scale{float(columns) / std::max(extent.x, 1e-20f), float(rows) / std::max(extent.y, 1e-20f),
                             float(resolution.z) / std::max(extent.z, 1e-20f)};
    const auto toCells = [&](const Core::float3 p) {
        return Core::float3{(p.x - bounds.min.x) * scale.x, (p.y - bounds.min.y) * scale.y, (p.z - bounds.min.z) * scale.z};
    };
    std::vector<std::vector<float>> crossings(columns * rows);
    for (std::size_t index = 0; index + 2 < indices.size(); index += 3) {
        const Core::float3 a = toCells(positions[indices[index]]);
        const Core::float3 b = toCells(positions[indices[index + 1]]);
        const Core::float3 c = toCells(positions[indices[index + 2]]);
        const float area = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (area == 0.0f)
            continue;
        const auto range = [](const float p, const float q, const float r, const std::size_t count) {
            //! The half offset lies off the usual grid lines of the vertices, so a column seldom meets an edge.
            const float low = std::floor(std::min({p, q, r}) - 1.0f);
            const float high = std::ceil(std::max({p, q, r}) + 1.0f);
            return std::pair{std::size_t(std::clamp(low, 0.0f, float(count))), std::size_t(std::clamp(high, 0.0f, float(count)))};
        };
        const auto [x0, x1] = range(a.x, b.x, c.x, columns);
        const auto [y0, y1] = range(a.y, b.y, c.y, rows);
        for (std::size_t y = y0; y < y1; ++y)
            for (std::size_t x = x0; x < x1; ++x) {
                const float px = float(x) + 0.50137f;
                const float py = float(y) + 0.49873f;
                const float u = ((b.x - px) * (c.y - py) - (b.y - py) * (c.x - px)) / area;
                const float v = ((c.x - px) * (a.y - py) - (c.y - py) * (a.x - px)) / area;
                const float w = 1.0f - u - v;
                if (u < 0.0f || v < 0.0f || w < 0.0f)
                    continue;
                crossings[y * columns + x].push_back(u * a.z + v * b.z + w * c.z);
            }
    }
    std::vector<float> filled(std::size_t{resolution.x} * resolution.y * resolution.z, 0.0f);
    for (std::size_t y = 0; y < rows; ++y)
        for (std::size_t x = 0; x < columns; ++x) {
            std::vector<float>& values = crossings[y * columns + x];
            std::ranges::sort(values);
            for (std::size_t pair = 0; pair + 1 < values.size(); pair += 2)
                for (auto z = static_cast<std::size_t>(std::max(values[pair], 0.0f)); z < resolution.z && float(z) < values[pair + 1]; ++z) {
                    const float share = std::min(float(z + 1), values[pair + 1]) - std::max(float(z), values[pair]);
                    filled[(z * resolution.y + y / subsamples) * resolution.x + x / subsamples] += share / float(subsamples * subsamples);
                }
        }
    std::vector<std::uint16_t> mask(filled.size());
    std::ranges::transform(filled, mask.begin(), [](const float value) { return Core::toHalf(std::min(value, 1.0f)).bits; });
    return mask;
}

}
