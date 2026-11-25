#include "DiffRenderer/Scenes/Nifti.h"

#include <algorithm>
#include <cstring>
#include <format>
#include <memory>
#include <span>
#include <utility>

#include <zlib.h>

namespace DiffRenderer::Scenes {

namespace {

constexpr std::int32_t headerSize = 348;

enum class Datatype : std::int16_t { Uint8 = 2, Int16 = 4, Int32 = 8, Float32 = 16, Float64 = 64, Uint16 = 512 };

struct GzipClose {
    void operator()(gzFile file) const { gzclose(file); }
};

using GzipFile = std::unique_ptr<std::remove_pointer_t<gzFile>, GzipClose>;

template <class T>
T readField(const std::array<char, headerSize>& header, const std::size_t offset) {
    T value{};
    std::memcpy(&value, header.data() + offset, sizeof(T));
    return value;
}

bool readExactly(gzFile file, void* target, const std::size_t size) {
    return gzread(file, target, static_cast<unsigned>(size)) == static_cast<int>(size);
}

template <class T>
bool readAs(gzFile file, const float slope, const float intercept, const std::span<float> values) {
    std::vector<T> stored(values.size());
    if (!readExactly(file, stored.data(), stored.size() * sizeof(T)))
        return false;
    std::ranges::transform(stored, values.begin(), [&](const T value) { return static_cast<float>(value) * slope + intercept; });
    return true;
}

std::expected<bool, std::string> readValues(gzFile file, const Datatype datatype, const float slope, const float intercept, const std::span<float> values) {
    switch (datatype) {
    case Datatype::Uint8:
        return readAs<std::uint8_t>(file, slope, intercept, values);
    case Datatype::Int16:
        return readAs<std::int16_t>(file, slope, intercept, values);
    case Datatype::Uint16:
        return readAs<std::uint16_t>(file, slope, intercept, values);
    case Datatype::Int32:
        return readAs<std::int32_t>(file, slope, intercept, values);
    case Datatype::Float32:
        return readAs<float>(file, slope, intercept, values);
    case Datatype::Float64:
        return readAs<double>(file, slope, intercept, values);
    }
    return std::unexpected(std::format("the NIfTI datatype {} is not supported", std::to_underlying(datatype)));
}

}

std::expected<NiftiVolume, std::string> readNifti(const std::filesystem::path& path) {
    const GzipFile file(gzopen(path.c_str(), "rb"));
    if (!file)
        return std::unexpected(std::format("{} does not open", path.string()));
    std::array<char, headerSize> header{};
    if (!readExactly(file.get(), header.data(), header.size()) || readField<std::int32_t>(header, 0) != headerSize)
        return std::unexpected(std::format("{} has no little-endian NIfTI-1 header", path.string()));
    const auto dimensions = readField<std::array<std::int16_t, 8>>(header, 40);
    const auto datatype = readField<Datatype>(header, 70);
    const auto dataOffset = static_cast<z_off_t>(readField<float>(header, 108));
    const auto storedSlope = readField<float>(header, 112);
    const float slope = storedSlope == 0.0f ? 1.0f : storedSlope;
    const float intercept = storedSlope == 0.0f ? 0.0f : readField<float>(header, 116);
    if (dimensions[0] < 3 || dimensions[1] < 1 || dimensions[2] < 1 || dimensions[3] < 1)
        return std::unexpected(std::format("{} is not a 3D volume", path.string()));
    if (dataOffset < headerSize || gzseek(file.get(), dataOffset, SEEK_SET) < 0)
        return std::unexpected(std::format("{} has the invalid data offset {}", path.string(), dataOffset));
    NiftiVolume volume;
    volume.size = {static_cast<std::uint32_t>(dimensions[1]), static_cast<std::uint32_t>(dimensions[2]), static_cast<std::uint32_t>(dimensions[3])};
    volume.values.resize(std::size_t{volume.size[0]} * volume.size[1] * volume.size[2]);
    const std::expected<bool, std::string> read = readValues(file.get(), datatype, slope, intercept, volume.values);
    if (!read)
        return std::unexpected(std::format("{}: {}", path.string(), read.error()));
    if (!*read)
        return std::unexpected(std::format("{} is truncated", path.string()));
    return volume;
}

}
