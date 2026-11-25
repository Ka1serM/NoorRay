#pragma once

#include <memory>
#include <optional>
#include <string>

#include "NoorRay.Python/Blender/BlenderTextures.h"
#include "NoorRay.Python/Blender/Diagnostics.h"
#include "NoorRay.Python/Blender/GridSet.h"
#include "NoorRay.Python/Blender/NodeTreeReader.h"
#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay.Python/Blender/TextureTable.h"
#include "NoorRay.Python/Blender/VolumeReader.h"
#include "NoorRay/Materials/Nodes/Folder.h"
#include "NoorRay/Materials/Nodes/MaterialCompiler.h"
#include "NoorRay/Materials/Nodes/NodeAttributes.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Python {

//! Reads the node tree and settings of a Blender material and compiles them into a material graph and a fixed Principled material.
class MaterialReader {
public:
    MaterialReader(Reader& reader, Diagnostics& diagnostics, TextureTable& textures, Scene::Scene& scene, Materials::AttributeTables& attributeTables,
                   NodeTreeReader& trees, ImageSamples& images)
        : reader(reader), diagnostics(diagnostics), textures(textures), scene(scene), attributeTables(attributeTables), trees(trees), images(images),
          volumes(reader, diagnostics, folder) {}

    struct Result {
        //! The material holds its node tree.
        Scene::Material material;
        Materials::MaterialNeeds needs;
        //! The volume of a material with a volume shader, without its placement.
        std::optional<Scene::Volume> volume;
    };

    Result read(const Object& material);
    //! The volume shader of the tree of a material in an object with grids, or of a world.
    std::optional<Scene::Volume> readVolume(const Materials::NodeTree& tree, const std::string& name, const char* outputType, GridSet* grids);

private:
    Reader& reader;
    Diagnostics& diagnostics;
    TextureTable& textures;
    Scene::Scene& scene;
    Materials::AttributeTables& attributeTables;
    NodeTreeReader& trees;
    ImageSamples& images;
    //! One random sequence for the materials of a sync, as Blender data gives them.
    Materials::Folder folder;
    VolumeReader volumes;
};

}
