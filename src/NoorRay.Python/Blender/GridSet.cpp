#include "NoorRay.Python/Blender/GridSet.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace NoorRay::Python {

float HostGrid::sample(const Core::float3 gridPoint) const {
    const std::array point{gridPoint.x - 0.5f, gridPoint.y - 0.5f, gridPoint.z - 0.5f};
    const std::array sides{size.x, size.y, size.z};
    std::array<std::array<std::size_t, 2>, 3> cell{};
    std::array<float, 3> weight{};
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (point[axis] < -1.0f || point[axis] > float(sides[axis]))
            return 0.0f;
        const float base = std::floor(point[axis]);
        weight[axis] = point[axis] - base;
        for (std::size_t side = 0; side < 2; ++side)
            cell[axis][side] = static_cast<std::size_t>(std::clamp(base + float(side), 0.0f, float(sides[axis] - 1)));
    }
    float sum = 0.0f;
    for (std::size_t corner = 0; corner < 8; ++corner) {
        const std::array high{corner & 1, (corner >> 1) & 1, corner >> 2};
        float factor = 1.0f;
        for (std::size_t axis = 0; axis < 3; ++axis)
            factor *= high[axis] ? weight[axis] : 1.0f - weight[axis];
        if (factor > 0.0f)
            sum += factor * values[(cell[2][high[2]] * size.y + cell[1][high[1]]) * size.x + cell[0][high[0]]];
    }
    return sum;
}

const HostGrid* GridSet::find(const std::string& name) {
    if (name.empty())
        return nullptr;
    const auto [found, inserted] = grids.try_emplace(name);
    if (!inserted)
        return found->second ? &*found->second : nullptr;
    const Object read = reader.helper("volume_grid", std::array{source, reader.string(name)});
    if (reader.failed() || !read)
        return nullptr;
    if (PyUnicode_Check(read.get())) {
        diagnostics.unsupported(reader.text(read));
        return nullptr;
    }
    const std::vector<Object> parts = reader.items(read);
    if (parts.size() != 3)
        return nullptr;
    HostGrid grid;
    const Values size = reader.values(parts[0]);
    grid.size = {static_cast<std::uint32_t>(size.values[0]), static_cast<std::uint32_t>(size.values[1]), static_cast<std::uint32_t>(size.values[2])};
    grid.toObject = reader.matrix(parts[1]);
    grid.toGrid = grid.toObject.inverse();
    const std::span<const std::byte> bytes = reader.bytes(parts[2]);
    grid.values.resize(bytes.size() / sizeof(float));
    if (reader.failed() || grid.values.empty() || grid.values.size() != std::size_t{grid.size.x} * grid.size.y * grid.size.z)
        return nullptr;
    std::memcpy(grid.values.data(), bytes.data(), bytes.size());
    for (float& value : grid.values) {
        value = std::isfinite(value) ? value : 0.0f;
        grid.maximum = std::max(grid.maximum, value);
    }
    found->second = std::move(grid);
    return &*found->second;
}

}
