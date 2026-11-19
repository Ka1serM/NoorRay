#pragma once

#include <cstddef>
#include <expected>
#include <filesystem>
#include <memory>
#include <span>

#include "NoorRay/Core/Error.h"

namespace NoorRay::Core {

//! A read-only view of a whole file: memory-mapped on the desktop, read into memory in the browser.
class MappedFile {
public:
    MappedFile();
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    ~MappedFile();

    [[nodiscard]] std::expected<void, Error> open(const std::filesystem::path& path);
    std::span<const std::byte> getBytes() const { return bytes; }

private:
    struct Mapping;
    std::unique_ptr<Mapping> mapping;
    std::span<const std::byte> bytes;
};

}
