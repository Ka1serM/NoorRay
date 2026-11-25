#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

#include "NoorRay.Python/Blender/Diagnostics.h"
#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay.Python/Blender/SkyTexture.h"
#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Python {

//! Straight keeps the colours of the file and Premultiplied scales them by alpha, as Cycles does for a node whose Alpha output has no link.
enum class AlphaUse : std::uint8_t { Straight, Premultiplied };

//! The texture of an image, and for a tiled (UDIM) image the columns and rows of its atlas.
struct ImageTexture {
    Scene::TextureHandle texture;
    std::uint32_t columns = 1;
    std::uint32_t rows = 1;
};

//! The scene texture of each Blender image and colour space, decoded once from its file, its packed data or its pixels.
class TextureTable {
public:
    //! The largest side of an image in texels, as the texture limit of Cycles; 0 has no limit. A new limit forgets the images.
    bool setLimit(std::uint32_t texels);
    //! The image user picks the frame of a sequence. An image without data is magenta, as in Cycles, and a warning.
    ImageTexture get(Reader& reader, Diagnostics& diagnostics, Scene::Scene& scene, const Object& image, const Object& user, bool isColor,
                     AlphaUse alpha = AlphaUse::Straight, bool needsTexels = false);
    void forgetImage(std::uint32_t image);
    //! Removes the textures that nothing uses from the scene.
    void removeUnused(Scene::Scene& scene, const std::set<std::pair<std::uint32_t, std::uint32_t>>& used);
    //! The table of an IES profile, one per profile text.
    [[nodiscard]] std::expected<Scene::TextureHandle, Core::Error> ies(Scene::Scene& scene, const std::string& name, std::string_view profile);
    //! The table of a Sky Texture, one per setting.
    Scene::TextureHandle sky(Scene::Scene& scene, const std::string& name, const SkySettings& settings);
    void clear() {
        textures.clear();
        iesTables.clear();
        skyTables.clear();
    }

private:
    //! The session_uid of the image, the path of its frame, its colour space and the use of its alpha.
    using Key = std::tuple<std::uint32_t, std::string, std::string, AlphaUse>;

    std::map<Key, ImageTexture> textures;
    std::uint32_t limit = 0;
    std::map<std::size_t, Scene::TextureHandle> iesTables;
    std::map<SkySettings, Scene::TextureHandle> skyTables;
};

}
