#include "NoorRay.Python/Blender/MeshReader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <format>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>

#include "NoorRay.Python/Blender/CyclesHash.h"
#include "NoorRay.Python/Blender/Parallel.h"
#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay/Scene/MeshTopology.h"

namespace NoorRay::Python {

namespace {

static_assert(sizeof(Core::float3) == 3 * sizeof(float) && sizeof(Core::float2) == 2 * sizeof(float));

struct AttributeField {
    std::string_view type;
    const char* field;
    std::uint8_t components;
    bool integer;
};

constexpr AttributeField attributeFields[] = {
    {"FLOAT", "value", 1, false},  {"INT", "value", 1, true},  {"FLOAT_VECTOR", "vector", 3, false}, {"FLOAT2", "vector", 2, false},
    {"BOOLEAN", "value", 1, true}, {"INT8", "value", 1, true}, {"FLOAT_COLOR", "color", 4, false},   {"BYTE_COLOR", "color", 4, false},
};

//! An array that leaves its elements uninitialised, so the first write also faults its pages in.
template <class T>
class Scratch {
public:
    explicit Scratch(const std::size_t count = 0) : values(std::make_unique_for_overwrite<T[]>(count)), count(count) {}

    T& operator[](const std::size_t index) { return values[index]; }
    const T& operator[](const std::size_t index) const { return values[index]; }
    std::size_t size() const { return count; }
    T* begin() const { return values.get(); }
    T* end() const { return values.get() + count; }
    std::span<T> span() const { return {values.get(), count}; }

private:
    std::unique_ptr<T[]> values;
    std::size_t count;
};

template <class T>
bool sameBytes(const std::span<const T> a, const std::span<const T> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

std::span<float> floats(std::vector<Core::float3>& values) {
    return {reinterpret_cast<float*>(values.data()), values.size() * 3};
}

template <class T>
std::span<float> floats(const Scratch<T>& values) {
    return {reinterpret_cast<float*>(values.begin()), values.size() * sizeof(T) / sizeof(float)};
}

//! Blender splits a quad along its 1-3 diagonal when the 0-2 diagonal folds it over.
bool isQuadFlipped(const Core::float3 v0, const Core::float3 v1, const Core::float3 v2, const Core::float3 v3) {
    const Core::float3 d02 = v2 - v0;
    return Core::dot(Core::cross(v1 - v0, d02), Core::cross(v3 - v0, d02)) > 0.0f;
}

Scene::NormalSource toNormalSource(const std::string_view domain) {
    if (domain == "POINT")
        return Scene::NormalSource::Vertex;
    if (domain == "FACE")
        return Scene::NormalSource::Face;
    return Scene::NormalSource::Stored;
}

}

//! Cycles' attr_create_random_per_island and attr_create_pointiness, as a value per vertex.
std::vector<float> MeshReader::derivedValues(const Object& mesh, const std::string& stream, const std::span<const Core::float3> positions) {
    const std::size_t count = positions.size();
    const Object edges = reader.get(mesh, "edges");
    std::vector<std::array<std::uint32_t, 2>> pairs(reader.size(edges));
    reader.foreachGet(edges, "vertices", std::span(pairs.empty() ? nullptr : pairs.front().data(), pairs.size() * 2));
    std::vector<float> values(count, 0.0f);
    if (reader.failed())
        return values;
    if (stream == Materials::islandStream) {
        std::vector<std::uint32_t> parents(count);
        std::vector<std::uint32_t> ranks(count, 0);
        std::iota(parents.begin(), parents.end(), 0u);
        const auto find = [&](std::uint32_t vertex) {
            while (parents[vertex] != vertex)
                vertex = parents[vertex] = parents[parents[vertex]];
            return vertex;
        };
        for (const auto& [from, to] : pairs) {
            std::uint32_t first = find(from);
            std::uint32_t second = find(to);
            if (first == second)
                continue;
            if (ranks[first] < ranks[second])
                std::swap(first, second);
            parents[second] = first;
            ranks[first] += ranks[first] == ranks[second] ? 1u : 0u;
        }
        for (std::uint32_t vertex = 0; vertex < count; ++vertex)
            values[vertex] = CyclesHash::toFloat(CyclesHash::hash(find(vertex)));
        return values;
    }
    std::vector<std::uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0u);
    const auto key = [&](const std::uint32_t vertex) { return std::tuple(positions[vertex].x, positions[vertex].y, positions[vertex].z, vertex); };
    std::ranges::sort(order, {}, key);
    std::vector<std::uint32_t> unique(count);
    for (std::size_t i = 0; i < count; ++i)
        unique[order[i]] = i > 0 && positions[order[i]] == positions[order[i - 1]] ? unique[order[i - 1]] : order[i];
    for (auto& [from, to] : pairs) {
        from = unique[from];
        to = unique[to];
        if (to < from)
            std::swap(from, to);
    }
    std::ranges::sort(pairs);
    pairs.erase(std::ranges::unique(pairs).begin(), pairs.end());
    std::erase_if(pairs, [](const std::array<std::uint32_t, 2>& pair) { return pair[0] == pair[1]; });
    std::vector<Core::float3> normals(count);
    reader.foreachGet(reader.get(mesh, "vertex_normals"), "vector", floats(normals));
    std::vector<Core::float3> sums(count);
    std::vector<Core::float3> directions(count);
    std::vector<std::uint32_t> neighbours(count, 0);
    for (std::uint32_t vertex = 0; vertex < count; ++vertex)
        sums[unique[vertex]] = sums[unique[vertex]] + normals[vertex];
    for (const auto& [from, to] : pairs) {
        const Core::float3 direction = Core::normalize(positions[to] - positions[from]);
        directions[from] = directions[from] + direction;
        directions[to] = directions[to] - direction;
        ++neighbours[from];
        ++neighbours[to];
    }
    std::vector<float> raw(count, 0.0f);
    for (std::uint32_t vertex = 0; vertex < count; ++vertex)
        if (neighbours[vertex] > 0)
            raw[vertex] =
                std::acos(std::clamp(Core::dot(Core::normalize(sums[vertex]), directions[vertex] / float(neighbours[vertex])), -1.0f, 1.0f)) / Core::pi;
    std::vector<float> blurred = raw;
    for (const auto& [from, to] : pairs) {
        blurred[from] += raw[to];
        blurred[to] += raw[from];
    }
    for (std::uint32_t vertex = 0; vertex < count; ++vertex)
        values[vertex] = blurred[unique[vertex]] / float(neighbours[unique[vertex]] + 1);
    return values;
}

std::vector<Core::float2> MeshReader::cornerStream(const Object& attributes, const Materials::UvStream& stream, const std::size_t faceCount,
                                                   const std::span<const std::uint32_t> faceStarts, const std::span<const std::int32_t> cornerVertices) {
    const Object attribute = reader.findItem(attributes, stream.name);
    const std::string type = attribute ? reader.text(attribute, "data_type") : std::string();
    const std::string domain = attribute ? reader.text(attribute, "domain") : std::string();
    const auto field = std::ranges::find(attributeFields, type, &AttributeField::type);
    if (field == std::end(attributeFields) || (domain != "POINT" && domain != "CORNER" && domain != "FACE"))
        return {};
    const std::size_t components = field->components;
    if (stream.part == Materials::UvPart::High && components < 3)
        return {};
    const Object data = reader.get(attribute, "data");
    std::vector<float> values(reader.size(data) * components);
    if (type == "BOOLEAN") {
        const auto flags = std::make_unique<bool[]>(values.size());
        reader.foreachGet(data, field->field, std::span(flags.get(), values.size()));
        std::ranges::copy(flags.get(), flags.get() + values.size(), values.begin());
    } else if (field->integer) {
        std::vector<std::int32_t> integers(values.size());
        reader.foreachGet(data, field->field, std::span(integers));
        std::ranges::transform(integers, values.begin(), [](const std::int32_t value) { return float(value); });
    } else {
        reader.foreachGet(data, field->field, std::span(values));
    }
    const std::size_t expected = domain == "FACE" ? faceCount : domain == "CORNER" ? cornerVertices.size() : values.size() / components;
    if (reader.failed() || values.size() != expected * components)
        return {};
    std::vector<Core::float2> result(cornerVertices.size());
    const auto pick = [&](const std::size_t element) -> Core::float2 {
        const float* value = &values[element * components];
        const float y = components > 1 ? value[1] : 0.0f;
        const float z = components > 2 ? value[2] : 0.0f;
        const float w = components > 3 ? value[3] : 1.0f;
        if (stream.part == Materials::UvPart::Mean)
            return {components == 1 ? value[0] : (value[0] + y + z) / 3.0f, w};
        if (stream.part == Materials::UvPart::High)
            return {z, w};
        return {value[0], components == 1 ? value[0] : y};
    };
    if (domain == "FACE")
        parallelFor(faceCount, [&](std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t face = begin; face < end; ++face) {
                const std::size_t last = face + 1 < faceCount ? faceStarts[face + 1] : cornerVertices.size();
                std::fill(result.begin() + std::ptrdiff_t(faceStarts[face]), result.begin() + std::ptrdiff_t(last), pick(face));
            }
        });
    else
        parallelFor(result.size(), [&](std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t corner = begin; corner < end; ++corner)
                result[corner] = pick(domain == "POINT" ? std::size_t(cornerVertices[corner]) : corner);
        });
    return result;
}

Scene::Mesh MeshReader::read(const Object& mesh, const std::string& name, const MeshRequest& request, MeshLayout& layout) {
    layout.reused = false;
    layout.patches.clear();
    Scene::Mesh result;
    result.name = name;
    const Object attributes = reader.get(mesh, "attributes");
    const Object polygons = reader.get(mesh, "polygons");
    const std::size_t vertexCount = reader.size(reader.get(mesh, "vertices"));
    const std::size_t cornerCount = reader.size(reader.get(mesh, "loops"));
    const std::size_t faceCount = reader.size(polygons);
    if (reader.failed() || faceCount == 0)
        return result;

    std::vector<Core::float3> sourcePositions = std::move(layout.sourcePositions);
    sourcePositions.resize(vertexCount);
    reader.foreachGet(reader.get(reader.item(attributes, "position"), "data"), "vector", floats(sourcePositions));
    Scratch<std::int32_t> cornerVertices(cornerCount);
    reader.foreachGet(reader.get(reader.item(attributes, ".corner_vert"), "data"), "value", cornerVertices.span());
    std::vector<std::uint32_t> faceStarts(faceCount);
    reader.foreachGet(polygons, "loop_start", std::span(faceStarts));
    std::vector<std::int32_t> faceSlots(faceCount, 0);
    if (const Object slots = reader.findItem(attributes, "material_index"); slots && reader.text(slots, "domain") == "FACE")
        reader.foreachGet(reader.get(slots, "data"), "value", std::span(faceSlots));
    //! Cycles shades a flat face with its geometric normal, so face normals are not read.
    const Scene::NormalSource normalSource = toNormalSource(reader.text(mesh, "normals_domain"));
    const bool vertexNormals = normalSource == Scene::NormalSource::Vertex;
    const bool cornerNormals = normalSource == Scene::NormalSource::Stored;
    std::vector<Core::float3> normals = std::move(layout.sourceNormals);
    normals.resize(vertexNormals ? vertexCount : cornerNormals ? cornerCount : 0);
    if (!normals.empty())
        reader.foreachGet(reader.get(mesh, vertexNormals ? "vertex_normals" : "corner_normals"), "vector", floats(normals));
    if (reader.failed())
        return result;
    const auto faceEnd = [&](const std::size_t face) { return face + 1 < faceCount ? std::size_t(faceStarts[face + 1]) : cornerCount; };
    std::vector<std::uint8_t> badChunks(chunkCount(cornerCount), 0);
    parallelFor(cornerCount, [&](const std::size_t chunk, const std::size_t begin, const std::size_t end) {
        for (std::size_t corner = begin; corner < end; ++corner)
            if (cornerVertices[corner] < 0 || std::size_t(cornerVertices[corner]) >= vertexCount) {
                badChunks[chunk] = 1;
                return;
            }
    });
    if (std::ranges::find(badChunks, 1) != badChunks.end()) {
        reader.fail(std::format("mesh '{}' has a corner with no vertex", name));
        return result;
    }

    Scratch<std::uint32_t> triangleCorners;
    Scratch<std::uint32_t> triangleFaces;
    const std::size_t faceChunks = chunkCount(faceCount);
    std::vector<std::size_t> chunkTriangles(faceChunks, 0);
    std::vector<std::size_t> chunkLargest(faceChunks, 0);
    parallelFor(faceCount, [&](const std::size_t chunk, const std::size_t begin, const std::size_t end) {
        std::size_t triangles = 0;
        std::size_t largest = 0;
        for (std::size_t face = begin; face < end; ++face) {
            const std::size_t size = faceEnd(face) - faceStarts[face];
            triangles += size == 3 ? 1 : size == 4 ? 2 : 0;
            largest = std::max(largest, size);
        }
        chunkTriangles[chunk] = triangles;
        chunkLargest[chunk] = largest;
    });
    const bool ownTriangles = std::ranges::max(chunkLargest) <= 4;
    std::vector<std::uint8_t> quadFlips(ownTriangles ? faceCount : 0, 0);
    if (ownTriangles) {
        std::size_t total = 0;
        for (std::size_t& count : chunkTriangles)
            total += std::exchange(count, total);
        triangleCorners = Scratch<std::uint32_t>(total * 3);
        triangleFaces = Scratch<std::uint32_t>(total);
        parallelFor(faceCount, [&](const std::size_t chunk, const std::size_t begin, const std::size_t end) {
            std::size_t triangle = chunkTriangles[chunk];
            const auto emit = [&](const std::uint32_t face, const std::uint32_t a, const std::uint32_t b, const std::uint32_t c) {
                std::uint32_t* corners = &triangleCorners[triangle * 3];
                corners[0] = a;
                corners[1] = b;
                corners[2] = c;
                triangleFaces[triangle++] = face;
            };
            const auto position = [&](const std::uint32_t corner) { return sourcePositions[std::size_t(cornerVertices[corner])]; };
            for (std::size_t face = begin; face < end; ++face) {
                const std::uint32_t first = faceStarts[face];
                const std::size_t size = faceEnd(face) - first;
                if (size == 3) {
                    emit(std::uint32_t(face), first, first + 1, first + 2);
                    continue;
                }
                if (size != 4)
                    continue;
                if (isQuadFlipped(position(first), position(first + 1), position(first + 2), position(first + 3))) {
                    quadFlips[face] = 1;
                    emit(std::uint32_t(face), first, first + 1, first + 3);
                    emit(std::uint32_t(face), first + 1, first + 2, first + 3);
                } else {
                    emit(std::uint32_t(face), first, first + 1, first + 2);
                    emit(std::uint32_t(face), first, first + 2, first + 3);
                }
            }
        });
    } else {
        const Object triangles = reader.get(mesh, "loop_triangles");
        const std::size_t triangleCount = reader.size(triangles);
        triangleCorners = Scratch<std::uint32_t>(triangleCount * 3);
        triangleFaces = Scratch<std::uint32_t>(triangleCount);
        reader.foreachGet(triangles, "loops", triangleCorners.span());
        reader.foreachGet(triangles, "polygon_index", triangleFaces.span());
    }
    const std::size_t triangleCount = triangleFaces.size();
    if (reader.failed() || triangleCount == 0)
        return result;

    std::string renderUv;
    std::string tangentUv;
    for (const Object& layer : reader.items(reader.get(mesh, "uv_layers"))) {
        const std::string layerName = reader.text(layer, "name");
        if (renderUv.empty() || reader.flag(layer, "active_render"))
            renderUv = layerName;
        if (layerName == request.tangentMap)
            tangentUv = layerName;
    }
    if (tangentUv.empty())
        tangentUv = renderUv;
    //! Cycles ignores a tangent-space normal map on a mesh without a UV map, and so does the shader.
    const bool tangents = request.tangents && !renderUv.empty();
    const auto cornerUv = [&](const std::string& uv) {
        std::vector<Core::float2> values(cornerCount);
        reader.foreachGet(reader.get(reader.item(attributes, uv.c_str()), "data"), "vector",
                          std::span(reinterpret_cast<float*>(values.data()), cornerCount * 2));
        return values;
    };
    std::array<std::vector<Core::float2>, Scene::maxUvSets> cornerUvs;
    std::uint32_t uvAliases = 0;
    if (!renderUv.empty())
        cornerUvs[0] = cornerUv(renderUv);
    for (std::size_t set = 1; set < cornerUvs.size(); ++set) {
        const std::optional<Materials::UvStream>& stream = request.streams[set];
        if (stream && stream->part == Materials::UvPart::Low && stream->name == renderUv)
            uvAliases |= 1u << set;
        else if (stream && (stream->name == Materials::pointinessStream || stream->name == Materials::islandStream)) {
            const std::vector<float> values = derivedValues(mesh, stream->name, sourcePositions);
            cornerUvs[set].resize(cornerCount);
            for (std::size_t corner = 0; corner < cornerCount && !reader.failed(); ++corner)
                cornerUvs[set][corner] = {values[std::size_t(cornerVertices[corner])], 1.0f};
        } else if (stream)
            cornerUvs[set] = cornerStream(attributes, *stream, faceCount, faceStarts, cornerVertices.span());
    }
    std::vector<std::uint32_t> packedColors;
    bool pointColors = false;
    const Object colorAttributes = request.colors ? reader.get(mesh, "color_attributes") : Object();
    const int renderIndex = colorAttributes ? reader.integer(colorAttributes, "render_color_index") : -1;
    const Object color =
        renderIndex >= 0 && std::size_t(renderIndex) < reader.size(colorAttributes) ? reader.at(colorAttributes, std::size_t(renderIndex)) : Object();
    //! Cycles reads 0 for a colour attribute that the mesh does not have.
    if (request.colors && !color)
        packedColors.assign(cornerCount, 0u);
    if (color) {
        const std::string domain = reader.text(color, "domain");
        if (domain != "POINT" && domain != "CORNER") {
            reader.fail(std::format("the colour attribute of mesh '{}' is on the domain {}", name, domain));
            return result;
        }
        pointColors = domain == "POINT";
        const Object data = reader.get(color, "data");
        std::vector<float> colors(reader.size(data) * 4);
        reader.foreachGet(data, "color", std::span(colors));
        const std::size_t sourceCount = pointColors ? vertexCount : cornerCount;
        if (colors.size() < sourceCount * 4) {
            reader.fail(std::format("the colour attribute of mesh '{}' is shorter than its domain", name));
            return result;
        }
        packedColors.resize(sourceCount);
        parallelFor(sourceCount, [&](std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t source = begin; source < end; ++source) {
                const float* value = &colors[source * 4];
                packedColors[source] = Core::packUnorm4x8({value[0], value[1], value[2], value[3]});
            }
        });
    }
    if (reader.failed())
        return result;

    //! Tangents and attributes do not follow new positions here, so a mesh with them reads in full.
    const bool reusable = ownTriangles && !tangents && !request.attributes;
    const bool sameUvs =
        layout.uvAliases == uvAliases && std::ranges::equal(cornerUvs, layout.uvs, [](const auto& a, const auto& b) { return sameBytes<Core::float2>(a, b); });
    if (reusable && !layout.triangleCorners.empty() && layout.normalSource == normalSource && layout.vertexCount == vertexCount && sameUvs
        && layout.pointColors == pointColors && sameBytes<std::int32_t>(cornerVertices.span(), layout.cornerVertices)
        && sameBytes<std::uint32_t>(faceStarts, layout.faceStarts) && sameBytes<std::int32_t>(faceSlots, layout.faceSlots)
        && sameBytes<std::uint32_t>(packedColors, layout.colors)) {
        for (std::size_t face = 0; face < faceCount; ++face) {
            if (quadFlips[face] == layout.quadFlips[face])
                continue;
            const std::uint32_t c = faceStarts[face];
            const std::array<std::uint32_t, 6> corners =
                quadFlips[face] ? std::array{c, c + 1, c + 3, c + 1, c + 2, c + 3} : std::array{c, c + 1, c + 2, c, c + 2, c + 3};
            std::ranges::copy(corners, layout.triangleCorners.begin() + layout.faceTriangles[face]);
            layout.patches.push_back(layout.faceTriangles[face]);
        }
        layout.quadFlips = std::move(quadFlips);
        layout.sourcePositions = std::move(sourcePositions);
        layout.sourceNormals = std::move(normals);
        layout.reused = true;
        return result;
    }
    layout = {};

    //! The triangles in the order of their material slots, and the Blender corner of each triangle corner.
    std::vector<std::uint32_t> faceTriangles(reusable ? faceCount : 0, 0);
    std::vector<std::uint32_t> sourceCorners(triangleCount * 3);
    const std::size_t slotCount = std::max<std::size_t>(request.slotMaterials.size(), 1);
    const auto slotOf = [&](const std::uint32_t face) { return std::size_t(std::clamp<std::int32_t>(faceSlots[face], 0, std::int32_t(slotCount - 1))); };
    std::vector<std::size_t> chunkSlots(chunkCount(triangleCount) * slotCount, 0);
    parallelFor(triangleCount, [&](const std::size_t chunk, const std::size_t begin, const std::size_t end) {
        std::vector<std::size_t> counts(slotCount, 0);
        for (std::size_t triangle = begin; triangle < end; ++triangle)
            ++counts[slotOf(triangleFaces[triangle])];
        std::ranges::copy(counts, chunkSlots.begin() + std::ptrdiff_t(chunk * slotCount));
    });
    const std::size_t triangleChunks = chunkSlots.size() / slotCount;
    std::size_t placed = 0;
    for (std::size_t slot = 0; slot < slotCount; ++slot) {
        const std::size_t first = placed;
        for (std::size_t chunk = 0; chunk < triangleChunks; ++chunk)
            placed += std::exchange(chunkSlots[chunk * slotCount + slot], placed);
        if (placed > first)
            result.subsets.push_back({.firstIndex = std::uint32_t(first * 3),
                                      .indexCount = std::uint32_t((placed - first) * 3),
                                      .material = slot < request.slotMaterials.size() ? request.slotMaterials[slot] : Scene::MaterialHandle()});
    }
    parallelFor(triangleCount, [&](const std::size_t chunk, const std::size_t begin, const std::size_t end) {
        std::vector<std::size_t> next(chunkSlots.begin() + std::ptrdiff_t(chunk * slotCount), chunkSlots.begin() + std::ptrdiff_t((chunk + 1) * slotCount));
        for (std::size_t triangle = begin; triangle < end; ++triangle) {
            const std::size_t target = next[slotOf(triangleFaces[triangle])]++;
            if (reusable && (triangle == 0 || triangleFaces[triangle - 1] != triangleFaces[triangle]))
                faceTriangles[triangleFaces[triangle]] = std::uint32_t(target * 3);
            for (std::size_t corner = 0; corner < 3; ++corner)
                sourceCorners[target * 3 + corner] = triangleCorners[triangle * 3 + corner];
        }
    });

    const std::size_t corners = sourceCorners.size();
    result.positions = sourcePositions;
    result.indices.resize(corners);
    result.normalDomain = vertexNormals ? Scene::MeshDomain::Vertex : Scene::MeshDomain::Corner;
    result.normals.resize(vertexNormals ? vertexCount : cornerNormals ? corners : 0);
    result.uvDomain = Scene::MeshDomain::Corner;
    result.uvAliases = uvAliases;
    for (std::size_t set = 0; set < cornerUvs.size(); ++set)
        result.uvs[set].resize(cornerUvs[set].empty() ? 0 : corners);
    result.colorDomain = pointColors ? Scene::MeshDomain::Vertex : Scene::MeshDomain::Corner;
    if (request.colors && pointColors)
        result.colors = packedColors;
    else if (request.colors)
        result.colors.resize(corners);
    parallelFor(corners, [&](std::size_t, const std::size_t begin, const std::size_t end) {
        for (std::size_t corner = begin; corner < end; ++corner) {
            const std::uint32_t source = sourceCorners[corner];
            result.indices[corner] = std::uint32_t(cornerVertices[source]);
            if (cornerNormals)
                result.normals[corner] = Core::packOctahedral(normals[source]);
            for (std::size_t set = 0; set < cornerUvs.size(); ++set)
                if (!cornerUvs[set].empty())
                    result.uvs[set][corner] = Core::packHalf2(cornerUvs[set][source].x, cornerUvs[set][source].y);
            if (request.colors && !pointColors)
                result.colors[corner] = packedColors[source];
        }
    });
    if (vertexNormals)
        parallelFor(vertexCount, [&](std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t vertex = begin; vertex < end; ++vertex)
                result.normals[vertex] = Core::packOctahedral(normals[vertex]);
        });
    result.computeBounds();
    if (tangents) {
        std::vector<Core::float3> tangentNormals = vertexNormals ? normals : std::vector<Core::float3>(cornerNormals ? corners : 0);
        const std::vector<Core::float2> source = tangentUv == renderUv ? cornerUvs[0] : cornerUv(tangentUv);
        std::vector<Core::float2> uv(corners);
        for (std::size_t corner = 0; corner < corners; ++corner) {
            uv[corner] = source[sourceCorners[corner]];
            if (cornerNormals)
                tangentNormals[corner] = normals[sourceCorners[corner]];
        }
        result.tangentDomain = Scene::MeshDomain::Corner;
        const std::vector<Core::float4> values = Import::MeshBuilder::computeTangents(result.positions, result.indices, tangentNormals, result.normalDomain, uv,
                                                                                      Scene::MeshDomain::Corner, Scene::MeshDomain::Corner);
        result.tangents.reserve(corners);
        for (const Core::float4 tangent : values)
            result.tangents.push_back(Core::packTangent({tangent.x, tangent.y, tangent.z}, tangent.w));
    }
    if (reusable) {
        layout.normalSource = normalSource;
        layout.vertexCount = vertexCount;
        layout.cornerVertices.assign(cornerVertices.begin(), cornerVertices.end());
        layout.faceStarts = faceStarts;
        layout.faceSlots = faceSlots;
        layout.quadFlips = std::move(quadFlips);
        layout.faceTriangles = std::move(faceTriangles);
        layout.triangleCorners = std::move(sourceCorners);
        layout.uvs = std::move(cornerUvs);
        layout.uvAliases = uvAliases;
        layout.colors = std::move(packedColors);
        layout.pointColors = pointColors;
        layout.sourcePositions = std::move(sourcePositions);
        layout.sourceNormals = std::move(normals);
    }
    if (!request.attributes)
        return result;

    std::vector<std::uint32_t> faceOffsets(faceStarts.begin(), faceStarts.end());
    faceOffsets.push_back(std::uint32_t(cornerCount));
    auto topology = std::make_shared<Scene::MeshTopology>();
    if (std::expected<void, Core::Error> opened =
            topology->open(normalSource, std::vector<std::uint32_t>(cornerVertices.begin(), cornerVertices.end()), std::move(faceOffsets), result);
        !opened) {
        reader.fail(std::format("mesh '{}': {}", name, opened.error().message));
        return result;
    }
    result.topology = std::move(topology);
    for (const std::string& attributeName : *request.attributes) {
        const Object attribute = reader.findItem(attributes, attributeName);
        if (!attribute) {
            reader.fail(std::format("mesh '{}' has no attribute '{}', which its displacement reads", name, attributeName));
            return result;
        }
        const std::string domain = reader.text(attribute, "domain");
        const std::string type = reader.text(attribute, "data_type");
        const auto field = std::ranges::find(attributeFields, type, &AttributeField::type);
        if (domain != "POINT" || field == std::end(attributeFields)) {
            reader.fail(std::format("the attribute '{}' of mesh '{}' is a {} on the domain {}; a displacement reads float values on points", attributeName,
                                    name, type, domain));
            return result;
        }
        const Object data = reader.get(attribute, "data");
        Scene::MeshAttribute& values = result.attributes.emplace_back();
        values.name = attributeName;
        values.components = field->components;
        values.values.resize(reader.size(data) * field->components);
        if (field->integer) {
            std::vector<std::int32_t> integers(values.values.size());
            reader.foreachGet(data, field->field, std::span(integers));
            std::ranges::transform(integers, values.values.begin(), [](const std::int32_t value) { return float(value); });
        } else {
            reader.foreachGet(data, field->field, std::span(values.values));
        }
    }
    return result;
}

bool MeshReader::fill(const MeshLayout& layout, Scene::Mesh& mesh) {
    const std::size_t corners = layout.triangleCorners.size();
    const bool vertexNormals = layout.normalSource == Scene::NormalSource::Vertex;
    const bool cornerNormals = layout.normalSource == Scene::NormalSource::Stored;
    if (mesh.positions.size() != layout.vertexCount || mesh.indices.size() != corners
        || mesh.normals.size()
               != (vertexNormals   ? layout.vertexCount
                   : cornerNormals ? corners
                                   : 0))
        return false;
    for (const std::uint32_t first : layout.patches) {
        if (first + 6 > corners)
            return false;
        for (std::uint32_t corner = first; corner < first + 6; ++corner) {
            const std::uint32_t source = layout.triangleCorners[corner];
            mesh.indices[corner] = std::uint32_t(layout.cornerVertices[source]);
            for (std::size_t set = 0; set < layout.uvs.size(); ++set)
                if (!layout.uvs[set].empty())
                    mesh.uvs[set][corner] = Core::packHalf2(layout.uvs[set][source].x, layout.uvs[set][source].y);
            if (!mesh.colors.empty() && !layout.pointColors)
                mesh.colors[corner] = layout.colors[source];
        }
    }
    std::ranges::copy(layout.sourcePositions, mesh.positions.begin());
    if (vertexNormals || cornerNormals)
        parallelFor(mesh.normals.size(), [&](std::size_t, const std::size_t begin, const std::size_t end) {
            for (std::size_t index = begin; index < end; ++index)
                mesh.normals[index] = Core::packOctahedral(layout.sourceNormals[vertexNormals ? index : layout.triangleCorners[index]]);
        });
    return true;
}

}
