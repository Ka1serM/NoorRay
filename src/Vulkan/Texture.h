#pragma once

#include "Context.h"
#include "Image.h"
#include <string>

class Texture {
    std::string name;
    int width = 0;
    int height = 0;

    Image image;
    vk::UniqueSampler sampler;
    vk::DescriptorImageInfo descriptorInfo;

    void createSampler(Context& context);

public:
    Texture(Context& context, const std::string& filepath);
    Texture(Context& context, const std::string& name, const void* data, int width, int height, vk::Format format);
    
    const std::string& getName() const { return name; }
    const vk::DescriptorImageInfo& getDescriptorInfo() const { return descriptorInfo; }
    const Image& getImage() const { return image; }
    const vk::Sampler& getSampler() const { return sampler.get(); }
};
