#include "NoorRay/Format/BinaryWriter.h"

#include <array>

namespace NoorRay::Format {

void BinaryWriter::write(const std::string_view text) {
    write(std::uint64_t{text.size()});
    put(std::as_bytes(std::span(text)));
}

void BinaryWriter::align() {
    constexpr std::array<std::byte, arrayAlignment> zeros{};
    put(std::span(zeros).first((arrayAlignment - position % arrayAlignment) % arrayAlignment));
}

void BinaryWriter::put(const std::span<const std::byte> bytes) {
    stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    position += bytes.size();
}

}
