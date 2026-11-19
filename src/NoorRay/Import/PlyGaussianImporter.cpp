#include "NoorRay/Import/PlyGaussianImporter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string>
#include <vector>

#include "NoorRay/Core/MappedFile.h"
#include "NoorRay/Import/PlyFile.h"

namespace NoorRay::Import {

namespace {

//! The records converted at a time; the whole file at once would need about twice its size again in floats.
constexpr std::size_t batchSize = 1u << 16;

float sigmoid(const float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

std::uint64_t spreadBits(std::uint64_t value) {
    value &= 0x1fffffu;
    value = (value | value << 32) & 0x1f00000000ffffu;
    value = (value | value << 16) & 0x1f0000ff0000ffu;
    value = (value | value << 8) & 0x100f00f00f00f00fu;
    value = (value | value << 4) & 0x10c30c30c30c30c3u;
    value = (value | value << 2) & 0x1249249249249249u;
    return value;
}

//! The slot of each record along a Morton curve, so that the BVH chunks of a cloud and the Gaussians that one ray tests are close in space and memory.
std::vector<std::uint32_t> spatialSlots(const PlyFile& file, const PlyElement& vertex) {
    static constexpr std::array<std::string_view, 3> axes{"x", "y", "z"};
    const std::vector<float> positions = file.readFloats(vertex, axes);
    const std::size_t count = vertex.count;
    Core::Bounds bounds;
    for (std::size_t index = 0; index < count; ++index)
        bounds.grow(Core::float3{positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]});
    const Core::float3 extent = Core::max(bounds.max - bounds.min, Core::float3{1e-20f, 1e-20f, 1e-20f});
    const std::array<float, 3> origin{bounds.min.x, bounds.min.y, bounds.min.z};
    const std::array<float, 3> scale{2097151.0f / extent.x, 2097151.0f / extent.y, 2097151.0f / extent.z};
    struct Keyed {
        std::uint64_t code;
        std::uint32_t index;
    };
    std::vector<Keyed> keyed(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto cell = [&](const std::size_t axis) {
            return static_cast<std::uint64_t>(std::clamp((positions[index * 3 + axis] - origin[axis]) * scale[axis], 0.0f, 2097151.0f));
        };
        keyed[index] = {spreadBits(cell(0)) | spreadBits(cell(1)) << 1 | spreadBits(cell(2)) << 2, static_cast<std::uint32_t>(index)};
    }
    std::sort(keyed.begin(), keyed.end(), [](const Keyed& a, const Keyed& b) { return a.code < b.code; });
    std::vector<std::uint32_t> slots(count);
    for (std::size_t rank = 0; rank < count; ++rank)
        slots[keyed[rank].index] = static_cast<std::uint32_t>(rank);
    return slots;
}

}

bool PlyGaussianImporter::isGaussianPly(const std::span<const std::byte> bytes) {
    PlyFile file;
    if (!file.open(bytes))
        return false;
    const PlyElement* vertex = file.findElement("vertex");
    return vertex && vertex->findProperty("f_dc_0") != PlyFile::missing && vertex->findProperty("rot_0") != PlyFile::missing;
}

std::expected<Scene::GaussianCloud, Core::Error> PlyGaussianImporter::read(const std::span<const std::byte> bytes, std::string name) {
    PlyFile file;
    if (std::expected<void, Core::Error> opened = file.open(bytes); !opened)
        return std::unexpected(std::move(opened.error()));
    const PlyElement* vertex = file.findElement("vertex");
    if (!vertex || vertex->hasLists)
        return Core::fail(Core::ErrorCode::InvalidFile, "the Gaussian PLY file has no plain vertex element");
    static constexpr std::array<std::string_view, 14> required{"x",       "y",       "z",       "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                                               "scale_0", "scale_1", "scale_2", "rot_0",  "rot_1",  "rot_2",  "rot_3"};
    for (const std::string_view property : required)
        if (vertex->findProperty(property) == PlyFile::missing)
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("the Gaussian PLY file has no property '{}'", property));
    std::vector<std::string> restNames;
    while (vertex->findProperty(std::format("f_rest_{}", restNames.size())) != PlyFile::missing)
        restNames.push_back(std::format("f_rest_{}", restNames.size()));
    std::uint32_t degree = 0;
    while (degree < 3 && 3 * (Scene::getShCoefficientCount(degree + 1) - 1) <= restNames.size())
        ++degree;
    const std::uint32_t coefficients = Scene::getShCoefficientCount(degree);
    const std::size_t restPerChannel = restNames.size() / 3;

    const std::vector<std::string_view> restViews(restNames.begin(), restNames.end());

    Scene::GaussianCloud cloud;
    cloud.name = std::move(name);
    cloud.shDegree = degree;
    //! The 3DGS training space is COLMAP's (y down, z forward); a half turn about x makes it y up without rotating the SH.
    cloud.transform.rows[1][1] = -1.0f;
    cloud.transform.rows[2][2] = -1.0f;
    const std::size_t count = vertex->count;
    cloud.positions.resize(count);
    cloud.scales.resize(count);
    cloud.rotations.resize(count);
    cloud.opacities.resize(count);
    cloud.sh.resize(count * coefficients * 3);
    const std::vector<std::uint32_t> slots = spatialSlots(file, *vertex);
    std::vector<float> values;
    std::vector<float> rest;
    for (std::size_t record = 0; record < count; ++record) {
        const std::size_t local = record % batchSize;
        if (local == 0) {
            const std::size_t batch = std::min(batchSize, count - record);
            file.readFloats(*vertex, required, record, batch, values);
            file.readFloats(*vertex, restViews, record, batch, rest);
        }
        const std::size_t index = slots[record];
        const float* value = values.data() + local * required.size();
        cloud.positions[index] = {value[0], value[1], value[2]};
        cloud.opacities[index] = sigmoid(value[6]);
        cloud.scales[index] = {std::exp(value[7]), std::exp(value[8]), std::exp(value[9])};
        const Core::float4 rotation{value[11], value[12], value[13], value[10]};
        const float length = std::sqrt(rotation.x * rotation.x + rotation.y * rotation.y + rotation.z * rotation.z + rotation.w * rotation.w);
        cloud.rotations[index] = length > 0.0f ? rotation * (1.0f / length) : Core::float4{0.0f, 0.0f, 0.0f, 1.0f};
        std::uint16_t* sh = cloud.sh.data() + index * coefficients * 3;
        for (std::uint32_t channel = 0; channel < 3; ++channel)
            sh[channel] = Core::toHalf(value[3 + channel]).bits;
        const float* restValues = rest.data() + local * restNames.size();
        for (std::uint32_t coefficient = 1; coefficient < coefficients; ++coefficient)
            for (std::uint32_t channel = 0; channel < 3; ++channel)
                sh[coefficient * 3 + channel] = Core::toHalf(restValues[channel * restPerChannel + coefficient - 1]).bits;
    }
    return cloud;
}

std::expected<Scene::GaussianCloudHandle, Core::Error> PlyGaussianImporter::import(const std::filesystem::path& path, Scene::Scene& scene) {
    Core::MappedFile file;
    if (std::expected<void, Core::Error> opened = file.open(path); !opened)
        return std::unexpected(std::move(opened.error()));
    std::expected<Scene::GaussianCloud, Core::Error> cloud = read(file.getBytes(), path.stem().string());
    if (!cloud)
        return std::unexpected(std::move(cloud.error()));
    const Core::Transform transform = cloud->transform;
    std::string name = cloud->name;
    const Scene::GaussianCloudHandle added = scene.addGaussianCloud(std::move(*cloud));
    scene.addNode({std::move(name), transform, {}, added});
    return added;
}

}
