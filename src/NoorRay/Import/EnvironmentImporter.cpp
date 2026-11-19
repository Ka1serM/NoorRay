#include "NoorRay/Import/EnvironmentImporter.h"

#include <algorithm>
#include <cctype>
#include <string>

#include "NoorRay/Core/MappedFile.h"
#include "NoorRay/Materials/TextureCache.h"

namespace NoorRay::Import {

bool EnvironmentImporter::isSupported(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension == ".hdr" || extension == ".exr";
}

std::expected<void, Core::Error> EnvironmentImporter::import(const std::filesystem::path& path, Scene::Scene& scene) {
    Core::MappedFile file;
    if (std::expected<void, Core::Error> opened = file.open(path); !opened)
        return opened;
    std::expected<Scene::Texture, Core::Error> decoded =
        Materials::TextureCache::decode(path.filename().string(), file.getBytes(), Materials::ColorSpace::Linear);
    if (!decoded)
        return std::unexpected(std::move(decoded.error()));
    Scene::Environment environment = scene.getEnvironment();
    const Scene::TextureHandle previous = environment.texture;
    environment.texture = scene.addTexture(std::move(*decoded));
    environment.color = {1.0f, 1.0f, 1.0f};
    scene.setEnvironment(environment);
    if (scene.find(previous))
        scene.remove(previous);
    return {};
}

}
