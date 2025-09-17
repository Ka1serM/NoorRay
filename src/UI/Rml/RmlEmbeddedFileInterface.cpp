#include "RmlEmbeddedFileInterface.h"
#include <cstring>
#include <algorithm>

RmlEmbeddedFileInterface::RmlEmbeddedFileInterface() = default;

RmlEmbeddedFileInterface::~RmlEmbeddedFileInterface() = default;

void RmlEmbeddedFileInterface::RegisterFile(const Rml::String& path, const std::span<const std::byte> data) {
    embedded_files[path] = data;
}

void RmlEmbeddedFileInterface::RegisterFile(const Rml::String& path, const unsigned char* data, const size_t size) {
    const std::span data_span(reinterpret_cast<const std::byte*>(data), size);
    RegisterFile(path, data_span);
}

Rml::FileHandle RmlEmbeddedFileInterface::Open(const Rml::String& path) {
    const auto it = embedded_files.find(path);
    if (it == embedded_files.end())
        return 0; // RmlUi uses 0 (nullptr) for a failed handle

    // Create a new EmbeddedFile tracker on the heap. This struct will act as our file handle.
    auto file = new EmbeddedFile{it->second, 0};
    return reinterpret_cast<Rml::FileHandle>(file);
}

void RmlEmbeddedFileInterface::Close(const Rml::FileHandle file) {
    EmbeddedFile* embedded_file = reinterpret_cast<EmbeddedFile*>(file);
    delete embedded_file;
}

size_t RmlEmbeddedFileInterface::Read(void* buffer, const size_t size, const Rml::FileHandle file) {
    const auto embedded_file = reinterpret_cast<EmbeddedFile*>(file);
    
    const size_t bytes_remaining = embedded_file->data.size() - embedded_file->offset;
    const size_t bytes_to_read = std::min(size, bytes_remaining);

    if (bytes_to_read > 0) {
        // Copy data from our embedded buffer into the buffer RmlUi provided.
        memcpy(buffer, embedded_file->data.data() + embedded_file->offset, bytes_to_read);
        embedded_file->offset += bytes_to_read;
    }

    return bytes_to_read;
}

bool RmlEmbeddedFileInterface::Seek(const Rml::FileHandle file, const long offset, const int origin) {
    auto* embedded_file = reinterpret_cast<EmbeddedFile*>(file);
    long long new_offset = 0;
    switch (origin) {
        case SEEK_SET:
            new_offset = offset;
            break;
        case SEEK_CUR:
            new_offset = static_cast<long long>(embedded_file->offset) + offset;
            break;
        case SEEK_END:
            new_offset = static_cast<long long>(embedded_file->data.size()) + offset;
            break;
        default:
            return false;
    }

    // Ensure the new offset is within the valid bounds of our data.
    if (new_offset >= 0 && new_offset <= static_cast<long long>(embedded_file->data.size())) {
        embedded_file->offset = static_cast<size_t>(new_offset);
        return true;
    }

    return false;
}

size_t RmlEmbeddedFileInterface::Tell(const Rml::FileHandle file) {
    const EmbeddedFile* embedded_file = reinterpret_cast<EmbeddedFile*>(file);
    return embedded_file->offset;
}