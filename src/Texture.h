#pragma once

#include "Context.h"
#include "Image.h"
#include <vulkan/vulkan.hpp>
#include <string>

class Texture {
public:
    Texture(const Context& context, const std::string& filepath);

    const vk::DescriptorImageInfo& getDescriptorInfo() const { return descriptorInfo; }
    const vk::Sampler& getSampler() const { return *sampler; }
    const Image& getImage() const { return image; }

private:
    Image image;
    vk::UniqueSampler sampler;
    vk::DescriptorImageInfo descriptorInfo;
};