#include "NoorRay/Import/UsdImporter.h"

namespace NoorRay::Import {

std::expected<void, Core::Error> UsdImporter::import(const std::filesystem::path&, Scene::Scene&) {
    return Core::fail(Core::ErrorCode::Unsupported, "this build has no USD support");
}

}
