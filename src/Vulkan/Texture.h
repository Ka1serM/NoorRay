#pragma once

#include "Context.h"
#include "Image.h"
#include <string>
#include <memory>

class Texture {
public:
    Texture(Context& context, const std::string& hdrFilepath, vk::Format format);
    Texture(Context& context, const std::string& name, const void* rgbaData, int width, int height);
    Texture(Context& context, const std::string& filepath);
    
    void createSampler(Context& context);

    const vk::DescriptorImageInfo& getDescriptorInfo() const { return descriptorInfo; }
    const vk::Sampler& getSampler() const { return *sampler; }
    const Image& getImage() const { return image; }
    const std::string& getName() const { return name; }

private:
    std::string name;
    Image image;
    vk::UniqueSampler sampler;
    vk::DescriptorImageInfo descriptorInfo;
};