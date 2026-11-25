#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "NoorRay.Python/Blender/Diagnostics.h"
#include "NoorRay.Python/Blender/NodeTreeReader.h"
#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay.Python/Blender/TextureTable.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Python {

//! The texels of each Blender image on the grid of the folder, by its pointer, for one sync.
using ImageSamples = std::unordered_map<std::uintptr_t, std::vector<Core::float4>>;

//! Loads the textures of the nodes of one root tree from Blender, and keeps a record of each for its node.
class BlenderTextures : public Materials::NodeTextureSource {
public:
    //! worldSky gives the table of the Sky Texture of a world without a graph.
    using WorldSky = std::function<Scene::TextureHandle(const Object& node)>;

    BlenderTextures(Reader& reader, Diagnostics& diagnostics, TextureTable& textures, Scene::Scene& scene, NodeTreeReader& trees, ImageSamples& images,
                    WorldSky worldSky = {})
        : reader(reader), diagnostics(diagnostics), textures(textures), scene(scene), trees(trees), images(images), worldSky(std::move(worldSky)) {}

    std::expected<Materials::NodeTexture, std::string> load(const Materials::NodeTree& tree, std::uint32_t node, Materials::NodeTextureUse use) override;
    std::span<const Core::float4> samples(const Materials::NodeTree& tree, std::uint32_t node) override;
    //! Adds the records of the textures that the compiles loaded to their nodes; a compile keeps references into the trees, so this comes after it.
    void storeRecords();

private:
    std::expected<Materials::NodeTexture, std::string> loadIes(const Object& node);
    //! Passes a failure of the reader to the node compile, which then stops.
    std::unexpected<std::string> failed();

    Reader& reader;
    Diagnostics& diagnostics;
    TextureTable& textures;
    Scene::Scene& scene;
    struct Record {
        const Materials::NodeTree* tree = nullptr;
        std::uint32_t node = 0;
        Materials::NodeProperty property;
    };

    NodeTreeReader& trees;
    ImageSamples& images;
    WorldSky worldSky;
    std::vector<Record> records;
};

}
