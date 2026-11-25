#include "NoorRay.Python/Blender/TextureTable.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <format>
#include <functional>
#include <optional>
#include <span>
#include <vector>

#include "NoorRay/Core/MappedFile.h"
#include "NoorRay/Core/Math.h"
#include "NoorRay/Materials/IesProfile.h"
#include "NoorRay/Materials/TextureCache.h"

namespace NoorRay::Python {

namespace {

constexpr std::array<std::uint8_t, 4> missingTexel{255, 0, 255, 255};
constexpr std::uint32_t maxAtlasSize = 16384;

bool isSrgb(const std::string_view space) {
    return space == "sRGB";
}

//! The spaces whose texels are linear Rec.709 or data; the decoder turns sRGB to linear.
bool isDirect(const std::string_view space) {
    constexpr std::string_view direct[] = {"sRGB", "Non-Color", "Raw", "Linear", "Linear Rec.709", "Working Space", ""};
    return std::ranges::contains(direct, space);
}

Materials::ColorSpace toColorSpace(const std::string_view space) {
    return isSrgb(space) ? Materials::ColorSpace::Srgb : Materials::ColorSpace::NonColor;
}

std::vector<float> readFloats(const Scene::Texture& texture) {
    const std::size_t count = std::size_t{texture.width} * texture.height * 4;
    std::vector<float> texels(count);
    const bool isHalf = texture.format == NoorRHI::ImageFormat::Rgba16Float;
    const bool srgb = texture.format == NoorRHI::ImageFormat::Rgba8Srgb;
    for (std::size_t index = 0; index < count; ++index) {
        if (isHalf) {
            Core::half value;
            std::memcpy(&value.bits, texture.data.data() + index * sizeof(value.bits), sizeof(value.bits));
            texels[index] = Core::fromHalf(value);
            continue;
        }
        const float value = float(std::to_integer<std::uint8_t>(texture.data[index])) / 255.0f;
        texels[index] = srgb && index % 4 != 3 ? Materials::TextureCache::srgbToLinear(value) : value;
    }
    return texels;
}

void toSceneLinear(Reader& reader, const std::span<float> texels, const std::string& space) {
    reader.helper("to_scene_linear", std::array{reader.string(space), reader.view(texels)});
}

//! The texels that Blender holds: bytes in the colour space of the image, or scene linear floats.
std::optional<Scene::Texture> readPixels(Reader& reader, const Object& image, const std::string& name, const std::string& space) {
    const Values size = reader.values(image, "size");
    const auto width = static_cast<std::size_t>(size.values[0]);
    const auto height = static_cast<std::size_t>(size.values[1]);
    const auto channels = static_cast<std::size_t>(reader.integer(image, "channels"));
    if (reader.failed() || width == 0 || height == 0 || channels == 0)
        return std::nullopt;
    std::vector<float> source(width * height * channels);
    reader.foreachGet(reader.get(image, "pixels"), nullptr, std::span(source));
    if (reader.failed())
        return std::nullopt;
    std::vector<float> texels(width * height * 4);
    for (std::size_t y = 0; y < height; ++y)
        for (std::size_t x = 0; x < width; ++x) {
            const float* from = source.data() + ((height - 1 - y) * width + x) * channels;
            float* to = texels.data() + (y * width + x) * 4;
            for (std::size_t channel = 0; channel < 3; ++channel)
                to[channel] = from[channels < 3 ? 0 : channel];
            to[3] = channels == 4 ? from[3] : 1.0f;
        }
    source = {};
    const bool isFloat = reader.flag(image, "is_float");
    if (!isFloat && isDirect(space)) {
        std::vector<std::uint8_t> bytes(texels.size());
        std::ranges::transform(texels, bytes.begin(),
                               [](const float value) { return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f); });
        return Materials::TextureCache::fromRgba8(name, std::uint32_t(width), std::uint32_t(height), bytes, isSrgb(space));
    }
    if (!isFloat)
        toSceneLinear(reader, texels, space);
    return Materials::TextureCache::fromRgbaFloat(name, std::uint32_t(width), std::uint32_t(height), texels);
}

std::optional<Scene::Texture> decode(Reader& reader, const std::string& name, const std::span<const std::byte> bytes, const std::string& space) {
    std::expected<Scene::Texture, Core::Error> decoded = Materials::TextureCache::decode(name, bytes, toColorSpace(space));
    if (!decoded)
        return std::nullopt;
    if (isDirect(space))
        return std::move(*decoded);
    std::vector<float> texels = readFloats(*decoded);
    toSceneLinear(reader, texels, space);
    return Materials::TextureCache::fromRgbaFloat(name, decoded->width, decoded->height, texels);
}

std::string absolutePath(Reader& reader, const Object& image, const std::string& path) {
    const Object absolute = reader.get(reader.module("bpy.path"), "abspath");
    return reader.text(reader.call(absolute, reader.string(path), "library", reader.get(image, "library")));
}

std::string framePath(Reader& reader, const Object& image, const Object& user, const std::string& source) {
    if (source != "SEQUENCE" || !user)
        return absolutePath(reader, image, reader.text(image, "filepath"));
    return reader.text(reader.helper("frame_path", std::array{image, user}));
}

void replaceAll(std::string& text, const std::string_view token, const std::string& value) {
    for (std::size_t at = text.find(token); at != std::string::npos; at = text.find(token, at + value.size()))
        text.replace(at, token.size(), value);
}

//! Cycles samples the tile under the texture coordinate; the atlas holds each tile at its column and row, with row 0 at the bottom.
std::optional<Scene::Texture> readTiles(Reader& reader, const Object& image, const std::string& name, const std::string& space, const std::uint32_t limit,
                                        ImageTexture& result) {
    struct Tile {
        std::uint32_t column;
        std::uint32_t row;
        Scene::Texture texture;
    };
    std::vector<Tile> tiles;
    std::map<std::uint32_t, Object> packed;
    const std::vector<Object> numbers = reader.items(reader.get(image, "tiles"));
    const std::vector<Object> files = reader.items(reader.get(image, "packed_files"));
    for (std::size_t index = 0; index < std::min(numbers.size(), files.size()); ++index)
        if (const Object file = reader.find(files[index], "packed_file"))
            packed.emplace(static_cast<std::uint32_t>(reader.integer(numbers[index], "number")), reader.get(file, "data"));
    const std::string pattern = absolutePath(reader, image, reader.text(image, "filepath"));
    std::uint32_t tileSize = 1;
    bool isFloat = false;
    for (const Object& number : numbers) {
        const auto tile = static_cast<std::uint32_t>(std::max(reader.integer(number, "number"), 1001) - 1001);
        Tile entry{tile % 10, tile / 10, {}};
        std::string path = pattern;
        replaceAll(path, "<UDIM>", std::to_string(tile + 1001));
        replaceAll(path, "<UVTILE>", std::format("u{}_v{}", entry.column + 1, entry.row + 1));
        Core::MappedFile mapped;
        std::span<const std::byte> bytes;
        if (const auto found = packed.find(tile + 1001); found != packed.end())
            bytes = reader.bytes(found->second);
        else if (mapped.open(path))
            bytes = mapped.getBytes();
        if (reader.failed())
            return std::nullopt;
        std::optional<Scene::Texture> texture = bytes.empty() ? std::nullopt : decode(reader, name, bytes, space);
        if (!texture)
            continue;
        entry.texture = std::move(*texture);
        tileSize = std::max({tileSize, entry.texture.width, entry.texture.height});
        isFloat = isFloat || entry.texture.format == NoorRHI::ImageFormat::Rgba16Float;
        result.columns = std::max(result.columns, entry.column + 1);
        result.rows = std::max(result.rows, entry.row + 1);
        tiles.push_back(std::move(entry));
    }
    if (tiles.empty())
        return std::nullopt;
    const std::uint32_t side = std::min({tileSize, maxAtlasSize / std::max(result.columns, result.rows), limit == 0 ? tileSize : limit});
    const std::size_t width = std::size_t{side} * result.columns;
    const std::size_t height = std::size_t{side} * result.rows;
    const std::size_t texelSize = isFloat ? 8 : 4;
    Scene::Texture atlas{name, tiles.front().texture.format, std::uint32_t(width), std::uint32_t(height), {}, {}};
    if (isFloat)
        atlas.format = NoorRHI::ImageFormat::Rgba16Float;
    atlas.data.resize(width * height * texelSize);
    for (Tile& tile : tiles) {
        Scene::Texture& texture = tile.texture;
        if (isFloat && texture.format != NoorRHI::ImageFormat::Rgba16Float)
            texture = Materials::TextureCache::fromRgbaFloat(name, texture.width, texture.height, readFloats(texture));
        const std::size_t left = std::size_t{tile.column} * side;
        const std::size_t top = std::size_t{result.rows - 1 - tile.row} * side;
        for (std::size_t y = 0; y < side; ++y)
            for (std::size_t x = 0; x < side; ++x) {
                const std::size_t from = (y * texture.height / side) * texture.width + x * texture.width / side;
                std::memcpy(atlas.data.data() + ((top + y) * width + left + x) * texelSize, texture.data.data() + from * texelSize, texelSize);
            }
        texture = {};
    }
    atlas.levels.push_back({0, atlas.data.size(), atlas.width, atlas.height});
    return atlas;
}

//! Halves the texture with a box filter until its sides are within limit.
void shrink(Scene::Texture& texture, const std::uint32_t limit) {
    const auto halve = [&]<class T>(const auto& toFloat, const auto& fromFloat) {
        const std::uint32_t width = std::max(texture.width / 2, 1u);
        const std::uint32_t height = std::max(texture.height / 2, 1u);
        const auto* source = reinterpret_cast<const T*>(texture.data.data());
        std::vector<std::byte> data(std::size_t{width} * height * 4 * sizeof(T));
        auto* target = reinterpret_cast<T*>(data.data());
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x)
                for (std::uint32_t channel = 0; channel < 4; ++channel) {
                    float sum = 0.0f;
                    for (std::uint32_t corner = 0; corner < 4; ++corner) {
                        const std::uint32_t fromX = std::min(x * 2 + (corner & 1u), texture.width - 1);
                        const std::uint32_t fromY = std::min(y * 2 + (corner >> 1u), texture.height - 1);
                        sum += toFloat(source[(std::size_t{fromY} * texture.width + fromX) * 4 + channel]);
                    }
                    target[(std::size_t{y} * width + x) * 4 + channel] = fromFloat(sum * 0.25f);
                }
        texture.data = std::move(data);
        texture.width = width;
        texture.height = height;
    };
    while (std::max(texture.width, texture.height) > limit)
        if (texture.format == NoorRHI::ImageFormat::Rgba16Float)
            halve.operator()<Core::half>([](const Core::half value) { return Core::fromHalf(value); }, [](const float value) { return Core::toHalf(value); });
        else
            halve.operator()<std::uint8_t>([](const std::uint8_t value) { return float(value); },
                                           [](const float value) { return static_cast<std::uint8_t>(value + 0.5f); });
    texture.levels = {{0, texture.data.size(), texture.width, texture.height}};
}

enum class AlphaChange : std::uint8_t { None, Multiply, Divide, Opaque };

AlphaChange alphaChange(const std::string_view mode, const AlphaUse use) {
    if (mode == "NONE")
        return AlphaChange::Opaque;
    if (mode == "STRAIGHT" && use == AlphaUse::Premultiplied)
        return AlphaChange::Multiply;
    return mode == "PREMUL" && use == AlphaUse::Straight ? AlphaChange::Divide : AlphaChange::None;
}

//! Cycles multiplies the stored values, which for an sRGB image are the encoded bytes; true when a texel is not opaque.
bool changeAlpha(Scene::Texture& texture, const AlphaChange change) {
    const auto apply = [&]<class T>(const auto& toFloat, const auto& fromFloat) {
        bool translucent = false;
        std::array<T, 4> texel;
        for (std::size_t offset = 0; offset + sizeof(texel) <= texture.data.size(); offset += sizeof(texel)) {
            std::memcpy(&texel, texture.data.data() + offset, sizeof(texel));
            const float alpha = toFloat(texel[3]);
            if (alpha >= 1.0f)
                continue;
            translucent = true;
            if (change == AlphaChange::None || (change == AlphaChange::Divide && alpha <= 0.0f))
                continue;
            if (change == AlphaChange::Opaque)
                texel[3] = fromFloat(1.0f);
            else
                for (std::size_t channel = 0; channel < 3; ++channel)
                    texel[channel] = fromFloat(change == AlphaChange::Multiply ? toFloat(texel[channel]) * alpha : toFloat(texel[channel]) / alpha);
            std::memcpy(texture.data.data() + offset, &texel, sizeof(texel));
        }
        return translucent;
    };
    if (texture.format == NoorRHI::ImageFormat::Rgba16Float)
        return apply.operator()<Core::half>([](const Core::half value) { return Core::fromHalf(value); },
                                            [](const float value) { return Core::toHalf(value); });
    return apply.operator()<std::uint8_t>([](const std::uint8_t value) { return float(value) / 255.0f; },
                                          [](const float value) { return static_cast<std::uint8_t>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f); });
}

}

bool TextureTable::setLimit(const std::uint32_t texels) {
    if (limit == texels)
        return false;
    limit = texels;
    textures.clear();
    return true;
}

void TextureTable::forgetImage(const std::uint32_t image) {
    std::erase_if(textures, [&](const auto& entry) { return std::get<0>(entry.first) == image; });
}

void TextureTable::removeUnused(Scene::Scene& scene, const std::set<std::pair<std::uint32_t, std::uint32_t>>& used) {
    const auto unused = [&](const Scene::TextureHandle handle) { return !used.contains({handle.index, handle.generation}); };
    std::erase_if(textures, [&](const auto& entry) { return unused(entry.second.texture); });
    std::erase_if(iesTables, [&](const auto& entry) { return unused(entry.second); });
    std::erase_if(skyTables, [&](const auto& entry) { return unused(entry.second); });
    std::vector<Scene::TextureHandle> removed;
    for (std::uint32_t dense = 0; dense < scene.getTextures().size(); ++dense)
        if (unused(scene.getTextures().keyAt(dense)))
            removed.push_back(scene.getTextures().keyAt(dense));
    for (const Scene::TextureHandle handle : removed)
        scene.remove(handle);
}

ImageTexture TextureTable::get(Reader& reader, Diagnostics& diagnostics, Scene::Scene& scene, const Object& image, const Object& user, const bool isColor,
                               const AlphaUse alpha, const bool needsTexels) {
    const std::string name = reader.text(image, "name");
    const std::string source = reader.text(image, "source");
    const std::string space = isColor ? reader.text(reader.get(image, "colorspace_settings"), "name") : std::string("Non-Color");
    const bool file = source == "FILE" || source == "SEQUENCE";
    const std::string path = file ? framePath(reader, image, user, source) : std::string();
    if (reader.failed())
        return {};
    const AlphaChange change = alphaChange(reader.text(image, "alpha_mode"), alpha);
    const Key key{static_cast<std::uint32_t>(reader.integer(image, "session_uid")), path, space, alpha};
    if (const auto found = textures.find(key); found != textures.end()) {
        const Scene::Texture* known = scene.find(found->second.texture);
        if (known && (!needsTexels || !known->data.empty()))
            return found->second;
    }

    ImageTexture result;
    std::optional<Scene::Texture> texture;
    std::string problem;
    if (source == "TILED") {
        texture = readTiles(reader, image, name, space, limit, result);
        problem = "no tile of it has a readable file";
    } else {
        std::span<const std::byte> encoded;
        Core::MappedFile mapped;
        const Object packed = source == "FILE" ? reader.find(image, "packed_file") : Object();
        const Object data = packed ? reader.get(packed, "data") : Object();
        if (data)
            encoded = reader.bytes(data);
        else if (file && mapped.open(path))
            encoded = mapped.getBytes();
        if (reader.failed())
            return {};
        if (!encoded.empty())
            texture = decode(reader, name, encoded, space);
        if (!texture && (!encoded.empty() || !file))
            texture = readPixels(reader, image, name, space);
        problem = encoded.empty() && file ? std::format("there is no file at '{}'", path) : "it has no pixels";
    }
    if (reader.failed())
        return {};
    if (!texture) {
        diagnostics.warn(std::format("image '{}' is magenta, because {}", name, problem));
        texture = Materials::TextureCache::fromRgba8(name, 1, 1, missingTexel, false);
    }
    if (limit != 0 && source != "TILED")
        shrink(*texture, limit);
    const bool translucent = changeAlpha(*texture, change);
    result.texture = scene.addTexture(std::move(*texture));
    textures.insert_or_assign(key, result);
    if (!translucent)
        textures.insert_or_assign(Key{std::get<0>(key), path, space, alpha == AlphaUse::Straight ? AlphaUse::Premultiplied : AlphaUse::Straight}, result);
    return result;
}

std::expected<Scene::TextureHandle, Core::Error> TextureTable::ies(Scene::Scene& scene, const std::string& name, const std::string_view profile) {
    const std::size_t key = std::hash<std::string_view>{}(profile);
    if (const auto found = iesTables.find(key); found != iesTables.end())
        return found->second;
    std::expected<Materials::IesProfile, Core::Error> parsed = Materials::parseIes(profile);
    if (!parsed)
        return std::unexpected(std::move(parsed.error()));
    const Scene::TextureHandle texture = scene.addTexture(Materials::makeIesTexture(*parsed, name));
    iesTables.emplace(key, texture);
    return texture;
}

Scene::TextureHandle TextureTable::sky(Scene::Scene& scene, const std::string& name, const SkySettings& settings) {
    if (const auto found = skyTables.find(settings); found != skyTables.end())
        return found->second;
    const Scene::TextureHandle texture = scene.addTexture(makeSkyTexture(settings, name));
    skyTables.emplace(settings, texture);
    return texture;
}

}
