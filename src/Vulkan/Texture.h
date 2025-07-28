#pragma once

#include "Context.h"
#include "Image.h"
#include <string>
#include <vector>
#include <glm/glm.hpp>

class Texture {
private:
    std::string name;
    int width = 0;
    int height = 0;
    std::vector<float> pixels; // CPU-side pixel data (RGBA floats)

    Image image; // GPU-side image representation
    vk::UniqueSampler sampler; // Vulkan sampler for this texture
    vk::DescriptorImageInfo descriptorInfo; // Descriptor info for shader binding

    void createSampler(Context& context);

public:
    Texture(Context& context, const std::string& filepath);
    Texture(Context& context, const std::string& name, const void* data, int width, int height, vk::Format format);

    glm::vec4 sample(glm::vec2 uv) const;

    const std::string& getName() const { return name; }
    const vk::DescriptorImageInfo& getDescriptorInfo() const { return descriptorInfo; }
    const Image& getImage() const { return image; }
    const vk::Sampler& getSampler() const { return sampler.get(); }
};
