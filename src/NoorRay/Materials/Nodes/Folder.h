#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"

namespace NoorRay::Materials {

//! The maximum of a volume grid by name; nothing when the volume has no such grid.
using GridMaximum = std::function<std::optional<float>(const std::string&)>;

//! Approximates a shader input driven by a node chain by its mean over a grid of samples: images give their pixels, procedural textures random values.
class Folder {
public:
    //! The mean of the input as one or three components, or nothing when the chain has a node the folder does not know.
    std::optional<Core::float3> average(NodeTextureSource& source, SocketRef socket, std::size_t components);

    //! An input over the values of the grid it reads, from 0 to the maximum of the grid; without a grid it has no values and is its average.
    struct Ramp {
        std::string grid;
        std::vector<Core::float3> values;
    };
    //! Nothing when the chain has a node the folder does not know, or reads more than one grid.
    std::optional<Ramp> ramp(NodeTextureSource& source, SocketRef socket, std::size_t components, const GridMaximum* grids);

private:
    using Samples = std::vector<Core::float4>;

    Samples evaluate(SocketRef socket, std::size_t components);
    Samples socketDefault(SocketRef socket);
    Samples input(SocketRef socket);
    std::vector<float> scalar(NodeRef node, const char* name);
    Samples output(NodeRef node, SocketRef socket);
    Samples mix(NodeRef node);
    std::vector<float> noise(float spread);
    //! The samples of a volume attribute: 0 without such a grid, as in Cycles.
    Samples grid(const std::string& name);

    NodeTextureSource* textures = nullptr;
    const GridMaximum* grids = nullptr;
    std::string gridName;
    std::mt19937 random{7};
    bool unknown = false;
    NodeLinks links;
};

}
