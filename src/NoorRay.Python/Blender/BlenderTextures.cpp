#include "NoorRay.Python/Blender/BlenderTextures.h"

#include <algorithm>
#include <cmath>
#include <format>

#include "NoorRay.Python/Blender/SkyTexture.h"
#include "NoorRay/Core/MappedFile.h"

namespace NoorRay::Python {

namespace {

using Materials::NodeTextureUse;

float srgbToLinear(const float value) {
    return value <= 0.04045f ? value / 12.92f : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

}

std::unexpected<std::string> BlenderTextures::failed() {
    diagnostics.forNodes();
    return std::unexpected(reader.getError() ? reader.getError()->message : std::string());
}

std::expected<Materials::NodeTexture, std::string> BlenderTextures::load(const Materials::NodeTree& tree, const std::uint32_t node, const NodeTextureUse use) {
    const Object source = trees.find(tree, node);
    if (!source)
        return std::unexpected("has no texture");
    const std::string& name = tree.nodes[node].name;
    Materials::NodeTexture result;
    switch (use) {
    case NodeTextureUse::Premultiplied:
    case NodeTextureUse::Straight:
    case NodeTextureUse::Color:
    case NodeTextureUse::Data:
    case NodeTextureUse::Background: {
        const bool graph = use == NodeTextureUse::Premultiplied || use == NodeTextureUse::Straight;
        const AlphaUse alpha = use == NodeTextureUse::Premultiplied ? AlphaUse::Premultiplied : AlphaUse::Straight;
        const ImageTexture image =
            textures.get(reader, diagnostics, scene, reader.find(source, "image"), reader.find(source, "image_user"),
                         graph || use == NodeTextureUse::Color || use == NodeTextureUse::Background, alpha, use == NodeTextureUse::Background);
        result = {image.texture, image.columns, image.rows};
        break;
    }
    case NodeTextureUse::Ies: {
        const std::expected<Materials::NodeTexture, std::string> table = loadIes(source);
        if (!table)
            return table;
        result = *table;
        break;
    }
    case NodeTextureUse::Sky: {
        const SkySettings settings = readSkySettings(reader, source);
        if (reader.failed())
            return failed();
        result.texture = textures.sky(scene, name, settings);
        break;
    }
    case NodeTextureUse::WorldSky:
        result.texture = worldSky ? worldSky(source) : Scene::TextureHandle();
        break;
    }
    if (reader.failed())
        return failed();
    records.push_back({&tree, node, Materials::textureRecord(use, result)});
    return result;
}

std::expected<Materials::NodeTexture, std::string> BlenderTextures::loadIes(const Object& node) {
    std::string profile;
    if (reader.text(node, "mode") == "INTERNAL") {
        const Object text = reader.find(node, "ies");
        if (!text)
            return std::unexpected("has no IES text");
        profile = reader.text(reader.call(text, "as_string"));
    } else {
        const Object absolute = reader.get(reader.module("bpy.path"), "abspath");
        const std::string path =
            reader.text(reader.call(absolute, reader.get(node, "filepath"), "library", reader.get(reader.get(node, "id_data"), "library")));
        Core::MappedFile file;
        if (reader.failed())
            return failed();
        if (!file.open(path))
            return std::unexpected(std::format("has no IES file at '{}'", path));
        const std::span<const std::byte> bytes = file.getBytes();
        profile.assign(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    if (reader.failed())
        return failed();
    const std::expected<Scene::TextureHandle, Core::Error> table = textures.ies(scene, reader.text(node, "name"), profile);
    if (!table)
        return std::unexpected(table.error().message);
    return Materials::NodeTexture{*table, 1, 1};
}

std::span<const Core::float4> BlenderTextures::samples(const Materials::NodeTree& tree, const std::uint32_t node) {
    constexpr std::size_t side = Materials::foldSide;
    const Object image = reader.find(trees.find(tree, node), "image");
    if (!image)
        return {};
    const std::uintptr_t pointer = reader.pointer(image);
    auto found = images.find(pointer);
    if (found == images.end()) {
        const Values size = reader.values(image, "size");
        const auto width = static_cast<std::size_t>(size.values[0]);
        const auto height = static_cast<std::size_t>(size.values[1]);
        const auto channels = static_cast<std::size_t>(reader.integer(image, "channels"));
        if (width == 0 || height == 0 || channels == 0 || reader.failed())
            return {};
        std::vector<float> pixels(width * height * channels);
        reader.foreachGet(reader.get(image, "pixels"), nullptr, std::span(pixels));
        const bool srgb = !reader.flag(image, "is_float") && reader.text(reader.get(image, "colorspace_settings"), "name") == "sRGB";
        std::vector<Core::float4> values(side * side, {1.0f, 1.0f, 1.0f, 1.0f});
        for (std::size_t row = 0; row < side; ++row) {
            const std::size_t y = row * (height - 1) / (side - 1);
            for (std::size_t column = 0; column < side; ++column) {
                const std::size_t x = column * (width - 1) / (side - 1);
                Core::float4& value = values[row * side + column];
                const float* pixel = pixels.data() + (y * width + x) * channels;
                for (std::size_t channel = 0; channel < std::min<std::size_t>(channels, 4); ++channel)
                    (&value.x)[channel] = pixel[channel];
                if (channels < 3)
                    value.y = value.z = value.x;
                if (srgb)
                    value = {srgbToLinear(value.x), srgbToLinear(value.y), srgbToLinear(value.z), value.w};
            }
        }
        found = images.emplace(pointer, std::move(values)).first;
    }
    records.push_back({&tree, node, Materials::samplesRecord(found->second)});
    return found->second;
}

void BlenderTextures::storeRecords() {
    for (Record& record : records)
        trees.addRecord(*record.tree, record.node, std::move(record.property));
    records.clear();
}

}
