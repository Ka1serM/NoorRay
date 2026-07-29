#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

enum class TextureEncoding
{
    Linear8,
    Srgb8,
    Float16,
    Float32,
};

class Texture
{
    friend class Scene;

public:
    explicit Texture(const std::string& filepath,
        TextureEncoding encoding = TextureEncoding::Linear8);
    Texture(std::string name, const void* data, int width, int height,
        TextureEncoding encoding);
    Texture(std::string name, std::vector<uint8_t>&& data, int width,
        int height, TextureEncoding encoding);
    Texture(std::string name, std::vector<uint16_t>&& data, int width,
        int height, TextureEncoding encoding = TextureEncoding::Float16);
    Texture(std::string name, std::vector<float>&& data, int width,
        int height, TextureEncoding encoding = TextureEncoding::Float32);
    Texture(std::string name,
        std::shared_ptr<const std::vector<uint8_t>> data, int width,
        int height, TextureEncoding encoding);
    Texture(std::string name,
        std::shared_ptr<const std::vector<uint16_t>> data, int width,
        int height, TextureEncoding encoding = TextureEncoding::Float16);
    Texture(std::string name,
        std::shared_ptr<const std::vector<float>> data, int width,
        int height, TextureEncoding encoding = TextureEncoding::Float32);

    // Frees the decoded pixels once no material or environment refers to this
    // texture. The zero size also tells the renderer to skip the slot when it
    // rebuilds its CUDA texture objects.
    void releaseResources();

    const std::string& getName() const { return name; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getSceneIndex() const { return sceneIndex; }
    TextureEncoding getEncoding() const { return encoding; }
    // Preserve ordinary image assets as RGBA8. CUDA's normalized texture
    // fetch performs sRGB-to-linear conversion when requested, so neither CPU
    // staging nor device storage needs an expanded linear representation.
    bool usesByteStorage() const {
        return bytePixels && !bytePixels->empty();
    }
    bool usesHalfStorage() const {
        return halfPixels && !halfPixels->empty();
    }
    const std::vector<uint8_t>& getBytePixels() const;
    const std::vector<uint16_t>& getHalfPixels() const;
    const std::shared_ptr<const std::vector<uint8_t>>&
    getByteStorage() const { return bytePixels; }
    const std::shared_ptr<const std::vector<uint16_t>>&
    getHalfStorage() const { return halfPixels; }
    const std::shared_ptr<const std::vector<float>>&
    getFloatStorage() const { return floatPixels; }
    // Returns linear float pixels for CPU consumers such as environment-map
    // importance sampling. Material textures never pay for this lazy copy.
    const std::vector<float>& getPixels() const;

private:
    void validateStorageSize(size_t valueCount) const;

    std::string name;
    int width{};
    int height{};
    int sceneIndex{-1};
    TextureEncoding encoding{TextureEncoding::Linear8};
    std::shared_ptr<const std::vector<uint8_t>> bytePixels;
    std::shared_ptr<const std::vector<uint16_t>> halfPixels;
    std::shared_ptr<const std::vector<float>> floatPixels;
    mutable std::vector<float> expandedPixels;
};
