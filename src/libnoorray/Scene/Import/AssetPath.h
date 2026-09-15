#pragma once

#include <filesystem>
#include <string>

namespace noorray
{

// Scene files reference assets by path. A path that does not exist as given
// (relative to whatever the process working directory is) is retried against
// the asset directory, so scenes can say "tests/utah_teapot.obj" regardless of
// where the host runs from. There is no default: the library carries no
// build-machine path, and a host that wants the fallback sets it at startup.
void setAssetDirectory(std::filesystem::path directory);
const std::filesystem::path& assetDirectory();

// Returns `path` if it exists, else assetDirectory() / path if that exists,
// else `path` unchanged so the caller reports the original name.
std::filesystem::path resolveAssetPath(const std::string& path);

} // namespace noorray
