#pragma once

#include "Context.h"
#include "Image.h"
#include <string>
#include <memory>

class Texture {
public:
    Texture(const std::shared_ptr<Context> &context, const std::string& filepath);

    const vk::DescriptorImageInfo& getDescriptorInfo() const { return descriptorInfo; }
    const vk::Sampler& getSampler() const { return *sampler; }
    const Image& getImage() const { return image; }

private:
    Image image;
    vk::UniqueSampler sampler;
    vk::DescriptorImageInfo descriptorInfo;
};