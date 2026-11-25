#include "NoorRay/Core/CacheDirectory.h"

#include <cstdlib>

namespace NoorRay::Core {

std::expected<std::filesystem::path, Error> getCacheDirectory() {
#if defined(_WIN32)
    if (const char* local = std::getenv("LOCALAPPDATA"); local && *local)
        return std::filesystem::path(local) / "NoorRay" / "Cache";
    return fail(ErrorCode::InvalidState, "LOCALAPPDATA is not set, so NoorRay has no cache folder");
#else
    if (const char* cache = std::getenv("XDG_CACHE_HOME"); cache && *cache)
        return std::filesystem::path(cache) / "noorray";
    if (const char* home = std::getenv("HOME"); home && *home)
        return std::filesystem::path(home) / ".cache" / "noorray";
    return fail(ErrorCode::InvalidState, "neither XDG_CACHE_HOME nor HOME is set, so NoorRay has no cache folder");
#endif
}

}
