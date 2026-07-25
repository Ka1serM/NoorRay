#pragma once

#include <string>
#include <vector>

enum class TextureEncoding
{
    Linear8,
    Srgb8,
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

    // Frees the decoded pixels once no material or environment refers to this
    // texture. The zero size also tells the renderer to skip the slot when it
    // rebuilds its CUDA texture objects.
    void releaseResources();

    const std::string& getName() const { return name; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getSceneIndex() const { return sceneIndex; }
    const std::vector<float>& getPixels() const { return pixels; }

private:
    std::string name;
    int width{};
    int height{};
    int sceneIndex{-1};
    std::vector<float> pixels;
};
