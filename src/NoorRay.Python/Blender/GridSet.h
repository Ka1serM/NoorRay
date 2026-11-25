#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "NoorRay.Python/Blender/Diagnostics.h"
#include "NoorRay.Python/Blender/Reader.h"

namespace NoorRay::Python {

//! A dense grid of an OpenVDB file or a fluid domain, x fastest; toObject places the corner of voxel 0 at the origin.
struct HostGrid {
    Core::uint3 size;
    Core::Transform toObject;
    Core::Transform toGrid;
    std::vector<float> values;
    float maximum = 0.0f;

    float sample(Core::float3 gridPoint) const;
};

//! The named grids of one volume object, read on first use.
class GridSet {
public:
    //! source is the path of an OpenVDB file or the object of a fluid domain.
    GridSet(Reader& reader, Diagnostics& diagnostics, Object source) : reader(reader), diagnostics(diagnostics), source(std::move(source)) {}

    //! Null when the object has no such grid.
    const HostGrid* find(const std::string& name);

private:
    Reader& reader;
    Diagnostics& diagnostics;
    Object source;
    std::map<std::string, std::optional<HostGrid>> grids;
};

}
