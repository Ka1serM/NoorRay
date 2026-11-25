#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "NoorRay/Materials/Nodes/Folder.h"
#include "NoorRay/Materials/Nodes/GraphCompiler.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeAttributes.h"
#include "NoorRay/Materials/Nodes/NodeDiagnostics.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Materials {

//! What a material needs from the meshes that use it.
struct MaterialNeeds {
    bool hasSurface = true;
    bool tangents = false;
    bool vertexColors = false;
    UvStreams streams;
    std::string tangentMap;
    bool displaces = false;
    std::vector<std::string> attributes;

    bool operator==(const MaterialNeeds&) const = default;
};

struct CompiledMaterial {
    Scene::Material material;
    MaterialNeeds needs;
};

//! Turns the node tree of a Blender material into a material graph and the fixed Principled material of NoorRay, with its displacement as a material program.
class MaterialCompiler {
public:
    MaterialCompiler(NodeDiagnostics& diagnostics, NodeTextureSource& textures, AttributeTables* attributeTables, Folder& folder)
        : diagnostics(diagnostics), textures(textures), attributeTables(attributeTables), folder(folder) {}

    //! The properties of the tree give the settings of the material: use_nodes, pass_index, displacement_method, emission_sampling, diffuse_color, metallic and roughness.
    CompiledMaterial compile(const NodeTree& tree, const std::string& name);

private:
    struct Record {
        CompiledMaterial result;
        //! The fixed inputs report to quiet diagnostics when a graph renders the material.
        NodeDiagnostics* diagnostics = nullptr;
    };

    void surface(Record& record, NodeRef node);
    void principled(Record& record, NodeRef node);
    void approximate(Record& record, NodeRef node);
    void normalMap(Record& record, NodeRef node, const char* name);
    bool imageTexture(Record& record, Scene::TextureSlot slot, NodeRef node, SocketRef socket, bool srgb, std::uint32_t channel);
    bool scalarTexture(Record& record, Scene::TextureSlot slot, SocketRef socket);
    void uvMapping(Record& record, NodeRef imageNode, Scene::MaterialTexture& texture);
    void displacement(Record& record, const NodeTree& tree, NodeRef output, GraphCompiler::Result& graph);
    void unsupported(const Record& record, const std::string& message);

    NodeDiagnostics& diagnostics;
    NodeTextureSource& textures;
    AttributeTables* attributeTables;
    Folder& folder;
    NodeLinks links;
};

}
