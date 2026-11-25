#pragma once

#include <expected>
#include <filesystem>

#include "NoorRay/Core/Error.h"

namespace NoorRay::Core {

//! The folder of the NoorRay caches: noorray in XDG_CACHE_HOME, ~/.cache, or LOCALAPPDATA on Windows.
[[nodiscard]] std::expected<std::filesystem::path, Error> getCacheDirectory();

}
