#include "Texture.h"
#include <stdexcept>
#include <stb_image.h>
#include <iostream>
#include <string>
#include "Utils.h"

Texture::Texture(Context& context, const std::string& filepath)
    : image([&]() -> Image {
        int texWidth = 0, texHeight = 0, texChannels = 0;

        float* rawPixels = stbi_loadf(filepath.c_str(), &texWidth, &texHeight, &texChannels, 4); 
        if (!rawPixels)
            throw std::runtime_error("Failed to load texture (stbi_loadf returned null). File: " + filepath);

        if (texWidth <= 0 || texHeight <= 0) {
            stbi_image_free(rawPixels);
            throw std::runtime_error("Loaded texture has invalid dimensions (W=" + std::to_string(texWidth) + ", H=" + std::to_string(texHeight) + "). File: " + filepath);
        }

        this->width = texWidth;
        this->height = texHeight;
        
        Image img(context, rawPixels, texWidth, texHeight, vk::Format::eR32G32B32A32Sfloat); 

        stbi_image_free(rawPixels);

        return img;
    }())
{
    name = Utils::nameFromPath(filepath);
    createSampler(context);
}

Texture::Texture(Context& context, const std::string& name, const void* data, int width, int height, vk::Format format)
    : image(context, data, width, height, format), name(name)
{
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Texture constructor (raw data): Invalid dimensions provided (W=" + std::to_string(width) + ", H=" + std::to_string(height) + "). Name: " + name);
    
    this->width = width;
    this->height = height;

    createSampler(context);
}

void Texture::createSampler(Context& context)
{
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);
    samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);

    sampler = context.getDevice().createSamplerUnique(samplerInfo);
    descriptorInfo.setImageView(image.getImageView());
    descriptorInfo.setSampler(*sampler);
    descriptorInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}