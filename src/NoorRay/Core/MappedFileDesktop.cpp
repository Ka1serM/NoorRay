#include "NoorRay/Core/MappedFile.h"

#include <format>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace NoorRay::Core {

#if defined(_WIN32)

struct MappedFile::Mapping {
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE section = nullptr;
    const void* view = nullptr;

    ~Mapping() {
        if (view)
            UnmapViewOfFile(view);
        if (section)
            CloseHandle(section);
        if (file != INVALID_HANDLE_VALUE)
            CloseHandle(file);
    }
};

std::expected<void, Error> MappedFile::open(const std::filesystem::path& path) {
    auto opened = std::make_unique<Mapping>();
    opened->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (opened->file == INVALID_HANDLE_VALUE)
        return fail(ErrorCode::Io, std::format("cannot open {}", path.string()));
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(opened->file, &size) || size.QuadPart == 0)
        return fail(ErrorCode::Io, std::format("{} is empty or unreadable", path.string()));
    opened->section = CreateFileMappingW(opened->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!opened->section)
        return fail(ErrorCode::Io, std::format("cannot map {}", path.string()));
    opened->view = MapViewOfFile(opened->section, FILE_MAP_READ, 0, 0, 0);
    if (!opened->view)
        return fail(ErrorCode::Io, std::format("cannot map {}", path.string()));
    bytes = std::span(static_cast<const std::byte*>(opened->view), static_cast<std::size_t>(size.QuadPart));
    mapping = std::move(opened);
    return {};
}

#else

struct MappedFile::Mapping {
    void* address = nullptr;
    std::size_t size = 0;

    ~Mapping() {
        if (address)
            munmap(address, size);
    }
};

std::expected<void, Error> MappedFile::open(const std::filesystem::path& path) {
    const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
        return fail(ErrorCode::Io, std::format("cannot open {}", path.string()));
    struct stat status{};
    if (fstat(descriptor, &status) != 0 || status.st_size == 0) {
        close(descriptor);
        return fail(ErrorCode::Io, std::format("{} is empty or unreadable", path.string()));
    }
    auto opened = std::make_unique<Mapping>();
    opened->size = static_cast<std::size_t>(status.st_size);
    opened->address = mmap(nullptr, opened->size, PROT_READ, MAP_PRIVATE, descriptor, 0);
    close(descriptor);
    if (opened->address == MAP_FAILED) {
        opened->address = nullptr;
        return fail(ErrorCode::Io, std::format("cannot map {}", path.string()));
    }
    bytes = std::span(static_cast<const std::byte*>(opened->address), opened->size);
    mapping = std::move(opened);
    return {};
}

#endif

MappedFile::MappedFile() = default;
MappedFile::MappedFile(MappedFile&&) noexcept = default;
MappedFile& MappedFile::operator=(MappedFile&&) noexcept = default;
MappedFile::~MappedFile() = default;

}
