#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "NoorRay.Python/Blender/Diagnostics.h"
#include "NoorRay.Python/Blender/GridSet.h"
#include "NoorRay/Materials/Nodes/Folder.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Python {

//! Turns the volume shader of a material or world into a volume without its placement.
class VolumeReader {
public:
    VolumeReader(Reader& reader, Diagnostics& diagnostics, Materials::Folder& folder) : reader(reader), diagnostics(diagnostics), folder(folder) {}

    //! grids are those of a volume object, and null for a mesh or a world; textures loads the images of the tree of the shader.
    std::optional<Scene::Volume> read(Materials::NodeRef shader, const std::string& name, GridSet* grids, Materials::NodeTextureSource& textures);

private:
    //! A socket over the values of one grid: one value for a constant, or a table from 0 to the maximum of the grid.
    struct Field {
        const HostGrid* grid = nullptr;
        std::vector<Core::float3> values{{1.0f, 1.0f, 1.0f}};

        Core::float3 at(Core::float3 objectPoint) const;
        Core::float3 mean() const;
        bool isConstant() const { return !grid; }
    };
    struct Emission {
        Field color;
        Field strength;
        //! The kelvin of a blackbody, whose strength goes from that of 1 to that of its temperature with intensity; below zero for none.
        float blackbody = -1.0f;
        Field temperature;
        Field temperatureGrid;

        Core::float3 at(Core::float3 objectPoint) const;
        bool isConstant() const;
    };

    void shade(Materials::NodeRef node, float weight);
    void principled(Materials::NodeRef node, float weight);
    void phase(Materials::NodeRef node);
    void unsupported(const std::string& message);
    //! The socket as components values; a chain that does not depend on one grid gives its mean.
    Field field(Materials::SocketRef input, std::size_t components);
    Field gridField(const std::string& name);
    Core::float3 constant(Materials::SocketRef input, std::size_t components);
    //! The scale of a closure by the density input: a constant factor, or 1 after the density became the grid of the volume.
    float density(Materials::NodeRef node, Materials::SocketRef input, const Field& attribute);
    void bakeGrids();

    Reader& reader;
    Diagnostics& diagnostics;
    Materials::Folder& folder;
    Materials::NodeTextureSource* textures = nullptr;
    Materials::NodeLinks links;
    GridSet* grids = nullptr;
    Materials::GridMaximum gridMaximum;
    Scene::Volume volume;
    std::optional<Field> densityField;
    Field densityAttribute;
    std::vector<Emission> emissions;
    std::uint32_t closures = 0;
    bool scatters = false;
};

}
