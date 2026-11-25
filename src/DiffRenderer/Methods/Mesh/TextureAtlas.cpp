#include "DiffRenderer/Methods/Mesh/TextureAtlas.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <memory>
#include <utility>

#include <xatlas.h>

#include "DiffRenderer/Core/ImageFile.h"

namespace DiffRenderer::Methods {

using NoorRHI::float2;

namespace {

constexpr std::uint32_t chartPadding = 2;
constexpr int dilationPasses = 4;
constexpr std::size_t texelChannels = 9;

//! One map of the material, as write_mtl of nvdiffrec names and encodes it: kd in sRGB, ks raw and the normal map from [-1, 1].
struct MapFile {
    const char* suffix;
    std::size_t offset;
    Core::Image::Encoding encoding;
    float scale;
    float bias;
};

constexpr std::array<MapFile, 3> mapFiles{{{"_kd", 0, Core::Image::Encoding::Linear, 1.0f, 0.0f},
                                           {"_ks", 3, Core::Image::Encoding::Srgb, 1.0f, 0.0f},
                                           {"_n", 6, Core::Image::Encoding::Srgb, 0.5f, 0.5f}}};

struct AtlasDeleter {
    void operator()(xatlas::Atlas* atlas) const { xatlas::Destroy(atlas); }
};

float getEdgeFunction(const float2 a, const float2 b, const float2 point) {
    return (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
}

class TexelCanvas {
public:
    TexelCanvas(const std::uint32_t width, const std::uint32_t height)
        : width(width), height(height), texels(std::size_t{width} * height * 3, 0.0f), covered(std::size_t{width} * height, 0) {}

    void fillTriangle(const std::array<float2, 3>& corners, const std::array<Vector3, 3>& colors) {
        const float area = getEdgeFunction(corners[0], corners[1], corners[2]);
        if (area == 0.0f)
            return;
        const auto [lowX, highX] = getTexelRange(corners[0].x, corners[1].x, corners[2].x, width);
        const auto [lowY, highY] = getTexelRange(corners[0].y, corners[1].y, corners[2].y, height);
        for (int y = lowY; y <= highY; ++y)
            for (int x = lowX; x <= highX; ++x) {
                const float2 center{static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f};
                const float w0 = getEdgeFunction(corners[1], corners[2], center) / area;
                const float w1 = getEdgeFunction(corners[2], corners[0], center) / area;
                const float w2 = 1.0f - w0 - w1;
                if (w0 < 0.0f || w1 < 0.0f || w2 < 0.0f)
                    continue;
                store(static_cast<std::size_t>(y) * width + x, colors[0] * w0 + colors[1] * w1 + colors[2] * w2);
            }
    }

    void dilate() {
        std::vector<std::size_t> filled;
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x) {
                const std::size_t index = std::size_t{y} * width + x;
                if (covered[index])
                    continue;
                Vector3 sum;
                int count = 0;
                for (const auto [dx, dy] : std::array<std::array<int, 2>, 4>{{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}}) {
                    const std::int64_t nx = std::int64_t{x} + dx;
                    const std::int64_t ny = std::int64_t{y} + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                        continue;
                    const std::size_t neighbor = static_cast<std::size_t>(ny) * width + static_cast<std::size_t>(nx);
                    if (!covered[neighbor])
                        continue;
                    sum += load(neighbor);
                    ++count;
                }
                if (count == 0)
                    continue;
                setColor(index, sum * (1.0f / static_cast<float>(count)));
                filled.push_back(index);
            }
        for (const std::size_t index : filled)
            covered[index] = 1;
    }

    std::vector<float> takeTexels() { return std::move(texels); }

private:
    static std::pair<int, int> getTexelRange(const float a, const float b, const float c, const std::uint32_t size) {
        const int low = static_cast<int>(std::floor(std::min({a, b, c}) - 0.5f));
        const int high = static_cast<int>(std::ceil(std::max({a, b, c}) - 0.5f));
        return {std::max(low, 0), std::min(high, static_cast<int>(size) - 1)};
    }

    Vector3 load(const std::size_t index) const { return {texels[index * 3], texels[index * 3 + 1], texels[index * 3 + 2]}; }

    void setColor(const std::size_t index, const Vector3 color) {
        texels[index * 3] = color.x;
        texels[index * 3 + 1] = color.y;
        texels[index * 3 + 2] = color.z;
    }

    void store(const std::size_t index, const Vector3 color) {
        setColor(index, color);
        covered[index] = 1;
    }

    std::uint32_t width;
    std::uint32_t height;
    std::vector<float> texels;
    std::vector<std::uint8_t> covered;
};

}

std::expected<TextureAtlas, std::string> unwrapMesh(const SurfaceMesh& mesh, const std::uint32_t resolution) {
    const std::unique_ptr<xatlas::Atlas, AtlasDeleter> atlas(xatlas::Create());
    xatlas::MeshDecl declaration;
    declaration.vertexPositionData = mesh.positions.data();
    declaration.vertexCount = static_cast<std::uint32_t>(mesh.positions.size());
    declaration.vertexPositionStride = sizeof(Vector3);
    declaration.indexData = mesh.triangles.data();
    declaration.indexCount = static_cast<std::uint32_t>(mesh.triangles.size());
    declaration.indexFormat = xatlas::IndexFormat::UInt32;
    if (const xatlas::AddMeshError error = xatlas::AddMesh(atlas.get(), declaration); error != xatlas::AddMeshError::Success)
        return std::unexpected(std::format("the mesh does not unwrap: {}", xatlas::StringForEnum(error)));
    xatlas::Generate(atlas.get(), {}, {.padding = chartPadding, .resolution = resolution, .blockAlign = true});
    if (atlas->atlasCount != 1)
        return std::unexpected(std::format("the charts of the mesh fill {} textures of {} texels instead of one", atlas->atlasCount, resolution));
    const xatlas::Mesh& unwrapped = atlas->meshes[0];
    if (unwrapped.indexCount != mesh.triangles.size())
        return std::unexpected("the unwrapped mesh has other triangles than the mesh");
    TextureAtlas result{.width = atlas->width, .height = atlas->height};
    const float2 scale{1.0f / static_cast<float>(atlas->width), 1.0f / static_cast<float>(atlas->height)};
    result.uvs.reserve(unwrapped.vertexCount);
    for (const xatlas::Vertex& vertex : std::span(unwrapped.vertexArray, unwrapped.vertexCount))
        result.uvs.push_back({vertex.uv[0] * scale.x, vertex.uv[1] * scale.y});
    result.uvTriangles.assign(unwrapped.indexArray, unwrapped.indexArray + unwrapped.indexCount);
    return result;
}

std::vector<float> bakeVertexValues(const SurfaceMesh& mesh, const TextureAtlas& atlas, const std::span<const Vector3> values) {
    TexelCanvas canvas(atlas.width, atlas.height);
    const float2 size{static_cast<float>(atlas.width), static_cast<float>(atlas.height)};
    for (std::size_t corner = 0; corner < mesh.triangles.size(); corner += 3) {
        std::array<float2, 3> corners;
        std::array<Vector3, 3> colors;
        for (std::size_t index = 0; index < 3; ++index) {
            const float2 uv = atlas.uvs[atlas.uvTriangles[corner + index]];
            corners[index] = {uv.x * size.x, uv.y * size.y};
            colors[index] = values[mesh.triangles[corner + index]];
        }
        canvas.fillTriangle(corners, colors);
    }
    for (int pass = 0; pass < dilationPasses; ++pass)
        canvas.dilate();
    return canvas.takeTexels();
}

std::expected<void, std::string> saveTexturedObj(const SurfaceMesh& mesh, const TextureAtlas& atlas, const std::span<const float> texels,
                                                 const std::filesystem::path& path) {
    const std::string stem = path.stem().string();
    std::filesystem::path materialPath = path;
    materialPath.replace_extension(".mtl");
    for (const MapFile& map : mapFiles) {
        Core::Image image(atlas.width, atlas.height);
        image.setEncoding(map.encoding);
        const std::span<float> pixels = image.getPixels();
        for (std::size_t texel = 0; texel < atlas.getTexelCount(); ++texel)
            for (std::size_t channel = 0; channel < 3; ++channel)
                pixels[texel * 3 + channel] = texels[texel * texelChannels + map.offset + channel] * map.scale + map.bias;
        if (std::expected<void, std::string> saved = Core::savePng(image, path.parent_path() / std::format("{}{}.png", stem, map.suffix)); !saved)
            return saved;
    }
    std::ofstream material(materialPath);
    material << std::format("newmtl texture\nillum 4\nKd 1 1 1\nKs 0 0 0\nmap_Kd {0}_kd.png\nmap_Ks {0}_ks.png\nbump {0}_n.png\n", stem);
    if (!material)
        return std::unexpected(std::format("{} is not fully written", materialPath.string()));
    std::ofstream stream(path);
    if (!stream)
        return std::unexpected(std::format("{} does not open for writing", path.string()));
    stream << std::format("mtllib {}.mtl\nusemtl texture\n", stem);
    for (const Vector3 position : mesh.positions)
        stream << std::format("v {} {} {}\n", position.x, position.y, position.z);
    for (const float2 uv : atlas.uvs)
        stream << std::format("vt {} {}\n", uv.x, 1.0f - uv.y);
    for (std::size_t corner = 0; corner < mesh.triangles.size(); corner += 3)
        stream << std::format("f {}/{} {}/{} {}/{}\n", mesh.triangles[corner] + 1, atlas.uvTriangles[corner] + 1, mesh.triangles[corner + 1] + 1,
                              atlas.uvTriangles[corner + 1] + 1, mesh.triangles[corner + 2] + 1, atlas.uvTriangles[corner + 2] + 1);
    if (!stream)
        return std::unexpected(std::format("{} is not fully written", path.string()));
    return {};
}

}
