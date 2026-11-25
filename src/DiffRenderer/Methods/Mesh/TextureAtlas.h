#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "DiffRenderer/Methods/Mesh/SurfaceMesh.h"
#include "NoorRHI/Core/Types.h"

namespace DiffRenderer::Methods {

//! The charts of a surface mesh packed into one texture; corner c of the mesh has the coordinates uvs[uvTriangles[c]], with v down the rows.
struct TextureAtlas {
    std::vector<NoorRHI::float2> uvs;
    std::vector<std::uint32_t> uvTriangles;
    std::uint32_t width = 0;
    std::uint32_t height = 0;

    std::size_t getTexelCount() const { return std::size_t{width} * height; }
};

[[nodiscard]] std::expected<TextureAtlas, std::string> unwrapMesh(const SurfaceMesh& mesh, std::uint32_t resolution);
//! Rasterizes a value on each vertex into the texture, with three values per texel.
std::vector<float> bakeVertexValues(const SurfaceMesh& mesh, const TextureAtlas& atlas, std::span<const Vector3> values);
//! Writes the OBJ and a material library with the kd, ks and normal maps of nvdiffrec as PNGs beside it; texels holds the nine values of each texel.
[[nodiscard]] std::expected<void, std::string> saveTexturedObj(const SurfaceMesh& mesh, const TextureAtlas& atlas, std::span<const float> texels,
                                                               const std::filesystem::path& path);

}
