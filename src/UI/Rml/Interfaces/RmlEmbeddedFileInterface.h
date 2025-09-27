#pragma once

#include <RmlUi/Core/FileInterface.h>
#include <unordered_map>
#include <span>
#include <cstddef>

// Represents an "open file" in memory, tracking the data and current read position.
struct EmbeddedFile {
    std::span<const std::byte> data;
    size_t offset = 0;
};

// A custom Rml::FileInterface that serves files from in-memory buffers.
class RmlEmbeddedFileInterface : public Rml::FileInterface {
public:
    RmlEmbeddedFileInterface();
    virtual ~RmlEmbeddedFileInterface();

    /**
        @brief Registers an in-memory asset to be served by this interface.
        @param path The virtual path for the asset (e.g., "rml/Editor.html"). This is what you'll use in RML href attributes.
        @param data A std::span viewing the embedded data from a #embed directive.
    */
    void RegisterFile(const Rml::String& path, std::span<const std::byte> data);

    void RegisterFile(const Rml::String& path, const unsigned char* data, size_t size);
    
    // --- Overridden Rml::FileInterface virtual methods ---

    /// Opens a file by looking up its path in the registered assets.
    Rml::FileHandle Open(const Rml::String& path) override;

    /// Closes a file handle, releasing its memory.
    void Close(Rml::FileHandle file) override;

    /// Reads data from the in-memory buffer.
    size_t Read(void* buffer, size_t size, Rml::FileHandle file) override;

    /// Seeks to a position within the in-memory buffer.
    bool Seek(Rml::FileHandle file, long offset, int origin) override;

    /// Returns the current read position in the in-memory buffer.
    size_t Tell(Rml::FileHandle file) override;

private:
    // A map linking virtual paths to their corresponding in-memory data spans.
    std::unordered_map<Rml::String, std::span<const std::byte>> embedded_files;
};