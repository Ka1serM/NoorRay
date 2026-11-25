#include "NoorRay.Python/Blender/VolumeReader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <string_view>

#include "NoorRay.Python/Blender/Parallel.h"
#include "NoorRay/Materials/Nodes/GraphCompiler.h"
#include "NoorRay/Spectral/Blackbody.h"

namespace NoorRay::Python {

namespace {

using Materials::NodeRef;
using Materials::SocketRef;

//! The voxels of the grid along the longest side.
constexpr std::uint32_t gridSide = 256;

Core::float3 clampPositive(const Core::float3 value) {
    return Core::max(value, Core::float3{});
}

Core::float3 blackbodyTable(const float kelvin) {
    constexpr float coldest = 800.0f;
    constexpr float step = 20.0f;
    static const std::vector<Core::float3> table = [] {
        std::vector<Core::float3> colors;
        for (std::size_t entry = 0; entry <= 560; ++entry)
            colors.push_back(Spectral::blackbodyColor(coldest + step * float(entry)));
        return colors;
    }();
    const float position = (kelvin - coldest) / step;
    return table[static_cast<std::size_t>(std::clamp(position + 0.5f, 0.0f, float(table.size() - 1)))];
}

//! The pack of Common/Math.slang.
std::uint32_t packRgb9e5(const Core::float3 value) {
    constexpr float largest = 65408.0f;
    const std::array channels{std::clamp(value.x, 0.0f, largest), std::clamp(value.y, 0.0f, largest), std::clamp(value.z, 0.0f, largest)};
    const float peak = std::max({channels[0], channels[1], channels[2]});
    if (!(peak > 0.0f))
        return 0;
    int exponent = std::max(-16, static_cast<int>(std::floor(std::log2(peak)))) + 16;
    float scale = std::exp2(static_cast<float>(24 - exponent));
    if (std::floor(peak * scale + 0.5f) == 512.0f) {
        scale *= 0.5f;
        ++exponent;
    }
    std::uint32_t packed = static_cast<std::uint32_t>(exponent) << 27;
    for (std::size_t channel = 0; channel < 3; ++channel)
        packed |= static_cast<std::uint32_t>(channels[channel] * scale + 0.5f) << (9 * channel);
    return packed;
}

}

Core::float3 VolumeReader::Field::at(const Core::float3 objectPoint) const {
    if (!grid)
        return values.front();
    const float position = grid->sample(grid->toGrid.transformPoint(objectPoint)) / std::max(grid->maximum, 1e-20f) * float(values.size() - 1);
    const auto base = static_cast<std::size_t>(std::clamp(position, 0.0f, float(values.size() - 1)));
    const std::size_t next = std::min(base + 1, values.size() - 1);
    const float weight = std::clamp(position - float(base), 0.0f, 1.0f);
    return values[base] * (1.0f - weight) + values[next] * weight;
}

Core::float3 VolumeReader::Field::mean() const {
    Core::float3 sum;
    for (const Core::float3 value : values)
        sum = sum + value;
    return sum / float(values.size());
}

Core::float3 VolumeReader::Emission::at(const Core::float3 objectPoint) const {
    const Core::float3 base = clampPositive(color.at(objectPoint)) * std::max(strength.at(objectPoint).x, 0.0f);
    if (blackbody < 0.0f)
        return base;
    //! The Principled Volume of Cycles.
    constexpr float sigma = 5.670373e-8f * 1e-6f / Core::pi;
    const float kelvin = std::max(temperature.at(objectPoint).x * temperatureGrid.at(objectPoint).x, 0.0f);
    const float intensity = sigma * (1.0f + blackbody * (kelvin * kelvin * kelvin * kelvin - 1.0f));
    return base * blackbodyTable(kelvin) * std::max(intensity, 0.0f);
}

bool VolumeReader::Emission::isConstant() const {
    return color.isConstant() && strength.isConstant() && temperature.isConstant() && temperatureGrid.isConstant();
}

void VolumeReader::unsupported(const std::string& message) {
    diagnostics.unsupported(std::format("material '{}': {}", volume.name, message));
}

std::optional<Scene::Volume> VolumeReader::read(const NodeRef shader, const std::string& name, GridSet* grids, Materials::NodeTextureSource& textures) {
    this->grids = grids;
    this->textures = &textures;
    links = {};
    gridMaximum = [grids](const std::string& grid) -> std::optional<float> {
        const HostGrid* found = grids ? grids->find(grid) : nullptr;
        return found ? std::optional(found->maximum) : std::nullopt;
    };
    volume = {};
    volume.name = name;
    volume.absorption = {};
    volume.scattering = {};
    densityField.reset();
    densityAttribute = {};
    emissions.clear();
    closures = 0;
    scatters = false;
    shade(shader, 1.0f);
    if (closures > 1 && (volume.densityGraph || densityField))
        unsupported("its volume adds shaders to one with a linked density, whose density then scales all of them");
    if (densityField || std::ranges::any_of(emissions, [](const Emission& emission) { return !emission.isConstant(); }))
        bakeGrids();
    else
        for (const Emission& emission : emissions)
            volume.emissionRate = volume.emissionRate + emission.at({});
    if (reader.failed())
        return std::nullopt;
    return std::move(volume);
}

VolumeReader::Field VolumeReader::field(const SocketRef input, const std::size_t components) {
    Field result;
    if (!input)
        return result;
    if (links.linkedNode(input).node) {
        std::optional<Materials::Folder::Ramp> ramp = folder.ramp(*textures, input, components, grids ? &gridMaximum : nullptr);
        if (!ramp)
            unsupported(std::format("the input '{}' of a volume shader is linked to nodes that do not render in a volume, and takes its default", input->name));
        else if (ramp->grid.empty())
            result.values = {*folder.average(*textures, input, components)};
        else {
            result.grid = grids->find(ramp->grid);
            result.values = std::move(ramp->values);
        }
        if (ramp)
            return result;
    }
    const Materials::NodeValues value = Materials::socketValues(input);
    result.values = {value.count == 1 ? Core::float3{value.values[0], value.values[0], value.values[0]}
                                      : Core::float3{value.values[0], value.values[1], value.values[2]}};
    return result;
}

VolumeReader::Field VolumeReader::gridField(const std::string& name) {
    Field result;
    result.grid = grids ? grids->find(name) : nullptr;
    if (!result.grid)
        return result;
    const float maximum = result.grid->maximum;
    result.values = {{}, {maximum, maximum, maximum}};
    return result;
}

Core::float3 VolumeReader::constant(const SocketRef input, const std::size_t components) {
    const Field value = field(input, components);
    if (!value.isConstant())
        diagnostics.warn(std::format("material '{}': the input '{}' of a volume shader varies over the grid, and is its mean", volume.name, input->name));
    return value.mean();
}

float VolumeReader::density(const NodeRef node, const SocketRef input, const Field& attribute) {
    const bool linked = static_cast<bool>(links.linkedNode(input).node);
    if (linked && !grids) {
        if (volume.densityGraph) {
            unsupported("it links a second density, which renders as 1");
            return 1.0f;
        }
        Materials::GraphCompiler compiler(diagnostics.forNodes(), *textures, nullptr, volume.name);
        std::optional<Materials::GraphCompiler::Result> compiled = compiler.compileValue(node, "Density");
        diagnostics.settle();
        if (compiled) {
            volume.kind = Scene::VolumeKind::Grid;
            volume.densityGraph = std::move(compiled->graph);
            volume.densityParameters = std::move(compiled->parameters);
            volume.densityTextures = std::move(compiled->textures);
        }
        return 1.0f;
    }
    Field value = field(input, 1);
    if (value.isConstant() && attribute.isConstant())
        return std::max(value.values.front().x * attribute.values.front().x, 0.0f);
    if (densityField) {
        if (densityField->grid != value.grid || densityAttribute.grid != attribute.grid)
            unsupported("its volume reads a second density grid, which renders as the first");
        return 1.0f;
    }
    densityField = std::move(value);
    densityAttribute = attribute;
    return 1.0f;
}

void VolumeReader::shade(const NodeRef node, const float weight) {
    const std::string_view type = Materials::nodeType(node);
    if (reader.failed())
        return;
    if (type == "ADD_SHADER" || type == "MIX_SHADER") {
        const std::vector<SocketRef> sockets = Materials::inputs(node);
        const bool mixes = type == "MIX_SHADER" && sockets.size() == 3;
        const float factor = mixes ? std::clamp(constant(sockets[0], 1).x, 0.0f, 1.0f) : 0.0f;
        for (std::size_t index = mixes ? 1 : 0; index < sockets.size(); ++index) {
            const float share = !mixes ? weight : index == 1 ? weight * (1.0f - factor) : weight * factor;
            if (const Materials::NodeLinkEnd link = links.linkedNode(sockets[index]); link.node && share > 0.0f)
                shade(link.node, share);
        }
        return;
    }
    const auto input = [&](const char* name) { return Materials::inputNamed(node, name); };
    const Core::float3 one{1.0f, 1.0f, 1.0f};
    if (type == "EMISSION") {
        Emission& emission = emissions.emplace_back(Emission{.color = field(input("Color"), 3), .strength = field(input("Strength"), 1)});
        for (Core::float3& value : emission.strength.values)
            value = value * weight;
        return;
    }
    if (type == "PRINCIPLED_VOLUME") {
        principled(node, weight);
        return;
    }
    if (type == "VOLUME_COEFFICIENTS") {
        ++closures;
        volume.absorption = volume.absorption + clampPositive(constant(input("Absorption Coefficients"), 3)) * weight;
        volume.scattering = volume.scattering + clampPositive(constant(input("Scatter Coefficients"), 3)) * weight;
        emissions.push_back({.color = field(input("Emission Coefficients"), 3)});
        for (Core::float3& value : emissions.back().color.values)
            value = value * weight;
        phase(node);
        return;
    }
    if (type != "VOLUME_SCATTER" && type != "VOLUME_ABSORPTION") {
        unsupported(std::format("the volume shader '{}' ({}) is not supported", node->name, type));
        return;
    }
    ++closures;
    const float scale = density(node, input("Density"), {}) * weight;
    const Core::float3 color = constant(input("Color"), 3);
    if (type == "VOLUME_ABSORPTION") {
        volume.absorption = volume.absorption + clampPositive(one - color) * scale;
        return;
    }
    volume.scattering = volume.scattering + clampPositive(color) * scale;
    phase(node);
}

void VolumeReader::principled(const NodeRef node, const float weight) {
    const auto input = [&](const char* name) { return Materials::inputNamed(node, name); };
    const auto attribute = [&](const char* name) {
        const SocketRef socket = input(name);
        return grids && socket ? socket->text : std::string();
    };
    if (const std::string color = attribute("Color Attribute"); !color.empty() && grids->find(color))
        unsupported(std::format("the volume shader '{}' reads its colour from a grid, which does not render", node->name));
    ++closures;
    const float scale = density(node, input("Density"), gridField(attribute("Density Attribute"))) * weight;
    const Core::float3 one{1.0f, 1.0f, 1.0f};
    const Core::float3 color = clampPositive(constant(input("Color"), 3));
    const Core::float3 tint = clampPositive(constant(input("Absorption Color"), 3));
    //! Cycles' absorption of the Principled Volume: (1 - color)(1 - sqrt(absorption color)).
    const Core::float3 root{std::sqrt(tint.x), std::sqrt(tint.y), std::sqrt(tint.z)};
    volume.scattering = volume.scattering + color * scale;
    volume.absorption = volume.absorption + clampPositive(one - color) * clampPositive(one - root) * scale;
    Emission emission{.color = field(input("Emission Color"), 3), .strength = field(input("Emission Strength"), 1)};
    for (Core::float3& value : emission.strength.values)
        value = value * weight;
    emissions.push_back(std::move(emission));
    const float blackbody = constant(input("Blackbody Intensity"), 1).x;
    if (blackbody > 0.0f) {
        Emission hot{.color = field(input("Blackbody Tint"), 3), .blackbody = std::min(blackbody, 1.0f)};
        hot.strength.values = {{weight, weight, weight}};
        hot.temperature = field(input("Temperature"), 1);
        hot.temperatureGrid = gridField(attribute("Temperature Attribute"));
        emissions.push_back(std::move(hot));
    }
    phase(node);
}

void VolumeReader::phase(const NodeRef node) {
    const std::string& name = node->name;
    if (scatters)
        unsupported(std::format("the volume shader '{}' adds a second phase function, and only its phase renders", name));
    scatters = true;
    const auto number = [&](const char* input) { return constant(Materials::inputNamed(node, input), 1).x; };
    const std::string type = Materials::findProperty(node, "phase") ? Materials::text(node, "phase") : std::string("HENYEY_GREENSTEIN");
    volume.draineAlpha = 0.0f;
    if (type == "RAYLEIGH") {
        //! Rayleigh is Draine with g 0 and alpha 1: 3 (1 + cos^2) / (16 pi).
        volume.anisotropy = 0.0f;
        volume.draineAlpha = 1.0f;
        return;
    }
    if (type == "MIE") {
        //! The Henyey-Greenstein and Draine mix of Jendersie and d'Eon 2023 as one lobe of the same mean cosine.
        const float diameter = std::clamp(number("Diameter"), 5.0f, 50.0f);
        const float forward = std::exp(-0.0990567f / (diameter - 1.67154f));
        const float draine = std::exp(-2.20679f / (diameter + 3.91029f) - 0.428934f);
        const float share = std::exp(-0.599085f / (diameter - 0.641583f) - 0.665888f);
        volume.anisotropy = forward * (1.0f - share) + draine * share;
        diagnostics.warn(std::format("material '{}': the Mie phase renders as Henyey-Greenstein of the same mean cosine", volume.name));
        return;
    }
    if (type == "FOURNIER_FORAND") {
        //! Half of the light scatters within this mean cosine in the sea water that the phase describes.
        volume.anisotropy = std::clamp(1.0f - 2.0f * number("Backscatter"), 0.0f, 0.98f);
        diagnostics.warn(std::format("material '{}': the Fournier-Forand phase renders as Henyey-Greenstein", volume.name));
        return;
    }
    volume.anisotropy = number("Anisotropy");
    if (type == "DRAINE")
        volume.draineAlpha = number("Alpha");
}

void VolumeReader::bakeGrids() {
    std::vector<const HostGrid*> used;
    const auto use = [&](const Field& field) {
        if (field.grid && !std::ranges::contains(used, field.grid))
            used.push_back(field.grid);
    };
    if (densityField) {
        use(*densityField);
        use(densityAttribute);
    }
    const bool emits = std::ranges::any_of(emissions, [](const Emission& emission) { return !emission.isConstant(); });
    for (const Emission& emission : emissions) {
        use(emission.color);
        use(emission.strength);
        use(emission.temperature);
        use(emission.temperatureGrid);
    }
    if (used.empty())
        return;
    const HostGrid& first = *used.front();
    Core::Bounds bounds{{}, {float(first.size.x), float(first.size.y), float(first.size.z)}};
    for (const HostGrid* grid : used)
        bounds.grow((first.toGrid * grid->toObject).transformBounds({{}, {float(grid->size.x), float(grid->size.y), float(grid->size.z)}}));
    const Core::float3 low{std::floor(bounds.min.x + 1e-3f), std::floor(bounds.min.y + 1e-3f), std::floor(bounds.min.z + 1e-3f)};
    const Core::float3 extent{std::ceil(bounds.max.x - 1e-3f) - low.x, std::ceil(bounds.max.y - 1e-3f) - low.y, std::ceil(bounds.max.z - 1e-3f) - low.z};
    const float step = std::max(Core::maxComponent(extent) / float(gridSide), 1.0f);
    const auto voxels = [&](const float side) { return std::max(static_cast<std::uint32_t>(std::ceil(side / step)), 1u); };
    volume.resolution = {voxels(extent.x), voxels(extent.y), voxels(extent.z)};
    volume.kind = Scene::VolumeKind::Grid;
    volume.bounds = {{}, {float(volume.resolution.x), float(volume.resolution.y), float(volume.resolution.z)}};
    Core::Transform local;
    for (int axis = 0; axis < 3; ++axis) {
        local.rows[axis][axis] = step;
        local.rows[axis][3] = (&low.x)[axis];
    }
    volume.transform = first.toObject * local;
    const std::size_t slice = std::size_t{volume.resolution.x} * volume.resolution.y;
    volume.density.resize(slice * volume.resolution.z);
    if (emits)
        volume.emissionGrid.resize(volume.density.size());
    parallelFor(volume.density.size(), [&](std::size_t, const std::size_t begin, const std::size_t end) {
        for (std::size_t voxel = begin; voxel < end; ++voxel) {
            const std::size_t row = voxel % slice;
            const Core::float3 point = volume.transform.transformPoint(
                {float(row % volume.resolution.x) + 0.5f, float(row / volume.resolution.x) + 0.5f, float(voxel / slice) + 0.5f});
            const float density = densityField ? std::max(densityField->at(point).x * densityAttribute.at(point).x, 0.0f) : 1.0f;
            volume.density[voxel] = Core::toHalf(std::min(density, 65504.0f)).bits;
            if (!emits)
                continue;
            Core::float3 emission;
            for (const Emission& term : emissions)
                emission = emission + term.at(point);
            volume.emissionGrid[voxel] = packRgb9e5(emission);
        }
    });
    if (!emits)
        for (const Emission& emission : emissions)
            volume.emissionRate = volume.emissionRate + emission.at({});
}

}
