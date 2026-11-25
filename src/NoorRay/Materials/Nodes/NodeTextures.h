#pragma once

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Materials {

//! How a compile reads the image or table of a node; each use has its own texture.
enum class NodeTextureUse : std::uint8_t {
    //! A graph lookup whose Alpha output has no link, which scales the colours by alpha.
    Premultiplied,
    //! A graph lookup whose Alpha output has a link.
    Straight,
    //! A fixed slot of colours.
    Color,
    //! A fixed slot of data, such as roughness or a normal map.
    Data,
    //! The equirectangular image of a world, which the device also samples on the host.
    Background,
    Ies,
    Sky,
    //! The sky table of a world without a graph.
    WorldSky,
};

//! The texture of a node, and for a tiled (UDIM) image the columns and rows of its atlas.
struct NodeTexture {
    Scene::TextureHandle texture;
    std::uint32_t columns = 1;
    std::uint32_t rows = 1;
};

//! The texels of an image on the grid of the folder, in linear colour, row by row.
inline constexpr std::size_t foldSide = 64;

//! Loads the textures that the nodes of a compile read, on first use.
class NodeTextureSource {
public:
    virtual ~NodeTextureSource() = default;

    //! The error is the reason that the node has no texture, such as an IES node without a profile.
    virtual std::expected<NodeTexture, std::string> load(const NodeTree& tree, std::uint32_t node, NodeTextureUse use) = 0;
    //! The foldSide x foldSide texels of the image of the node; empty when the image has no pixels.
    virtual std::span<const Core::float4> samples(const NodeTree& tree, std::uint32_t node) = 0;
};

//! The records that a compile stores in the properties of a texture node, so that a later compile finds the same textures without Blender.
NodeProperty textureRecord(NodeTextureUse use, const NodeTexture& texture);
NodeProperty samplesRecord(std::span<const Core::float4> samples);
bool isTextureRecord(const NodeProperty& property);
//! Adds a record to a node, unless the node has a record of the same use.
void addTextureRecord(Node& node, NodeProperty record);

//! The textures that the records of the texture nodes hold.
class RecordedTextures : public NodeTextureSource {
public:
    std::expected<NodeTexture, std::string> load(const NodeTree& tree, std::uint32_t node, NodeTextureUse use) override;
    std::span<const Core::float4> samples(const NodeTree& tree, std::uint32_t node) override;

private:
    std::vector<std::vector<Core::float4>> texels;
};

}
