#pragma once

#include "Context.h"
#include <string>
#include <vector>

class Texture {
    std::string name;
    int width = 0;
    int height = 0;

    std::vector<float> pixels;

public:
    Texture(Context& context, const std::string& filepath, vk::Format format = vk::Format::eR8G8B8A8Unorm);
    Texture(Context& context, const std::string& name, const void* data, int width, int height, vk::Format format);
    
    const std::string& getName() const { return name; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    const std::vector<float>& getPixels() const { return pixels; }
};
