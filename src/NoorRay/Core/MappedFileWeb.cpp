#include "NoorRay/Core/MappedFile.h"

#include <format>
#include <fstream>
#include <vector>

namespace NoorRay::Core {

struct MappedFile::Mapping {
    std::vector<std::byte> data;
};

std::expected<void, Error> MappedFile::open(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return fail(ErrorCode::Io, std::format("cannot open {}", path.string()));
    auto opened = std::make_unique<Mapping>();
    opened->data.resize(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(opened->data.data()), static_cast<std::streamsize>(opened->data.size())))
        return fail(ErrorCode::Io, std::format("cannot read {}", path.string()));
    bytes = opened->data;
    mapping = std::move(opened);
    return {};
}

MappedFile::MappedFile() = default;
MappedFile::MappedFile(MappedFile&&) noexcept = default;
MappedFile& MappedFile::operator=(MappedFile&&) noexcept = default;
MappedFile::~MappedFile() = default;

}
