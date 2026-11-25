#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Error.h"

namespace NoorRay::OpenDlssRr {

//! Linux uses the libcurl of the system, which it loads at run time; Windows uses WinHTTP.
[[nodiscard]] std::expected<std::vector<std::byte>, Core::Error> downloadRange(std::string_view url, std::uint64_t offset, std::uint64_t size);

}
