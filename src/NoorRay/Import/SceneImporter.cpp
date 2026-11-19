#include "NoorRay/Import/SceneImporter.h"

#include <algorithm>
#include <cctype>
#include <format>
#include <string>

#include "NoorRay/Format/SceneFile.h"
#include "NoorRay/Import/EnvironmentImporter.h"
#include "NoorRay/Import/GltfImporter.h"
#include "NoorRay/Import/ObjImporter.h"
#include "NoorRay/Import/PbrtImporter.h"
#include "NoorRay/Import/PlyGaussianImporter.h"
#include "NoorRay/Import/UsdImporter.h"

namespace NoorRay::Import {

namespace {

std::string getExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(), [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return extension;
}

}

bool SceneImporter::isSupported(const std::filesystem::path& path) {
    const std::string extension = getExtension(path);
    return extension == ".nrs" || extension == ".gltf" || extension == ".glb" || extension == ".obj" || extension == ".ply" || extension == ".pbrt"
           || extension == ".usd" || extension == ".usda" || extension == ".usdc" || extension == ".usdz" || EnvironmentImporter::isSupported(path);
}

std::expected<void, Core::Error> SceneImporter::import(const std::filesystem::path& path, Scene::Scene& scene) {
    if (!std::filesystem::exists(path))
        return Core::fail(Core::ErrorCode::Io, std::format("the file '{}' does not exist", path.string()));
    const std::string extension = getExtension(path);
    if (extension == ".nrs") {
        Format::SceneFile file;
        if (std::expected<void, Core::Error> opened = file.open(path); !opened)
            return opened;
        return file.load(scene);
    }
    if (extension == ".gltf" || extension == ".glb")
        return GltfImporter::import(path, scene);
    if (extension == ".obj")
        return ObjImporter::import(path, scene);
    if (extension == ".pbrt")
        return PbrtImporter::import(path, scene);
    if (extension == ".ply") {
        std::expected<Scene::GaussianCloudHandle, Core::Error> cloud = PlyGaussianImporter::import(path, scene);
        if (!cloud)
            return std::unexpected(std::move(cloud.error()));
        return {};
    }
    if (extension == ".usd" || extension == ".usda" || extension == ".usdc" || extension == ".usdz")
        return UsdImporter::import(path, scene);
    if (EnvironmentImporter::isSupported(path))
        return EnvironmentImporter::import(path, scene);
    return Core::fail(Core::ErrorCode::Unsupported, std::format("the file type '{}' is not supported", extension));
}

}
