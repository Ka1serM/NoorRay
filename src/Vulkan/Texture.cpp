#include "Texture.h"
#include <stdexcept>
#include <stb_image.h>
#include <cstring>
#include <iostream>
#include <string>
#include <algorithm>

#include "Utils.h"

#ifndef EPSILON
#define EPSILON 1e-6f
#endif

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
        
        size_t expected_elements = static_cast<size_t>(texWidth) * texHeight * 4;

        this->pixels.assign(rawPixels, rawPixels + expected_elements);

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

    size_t expected_elements_for_pixels_vector = static_cast<size_t>(width) * height * 4;

    this->pixels.resize(expected_elements_for_pixels_vector);

    if (format == vk::Format::eR32G32B32A32Sfloat) {
        size_t data_bytes_to_copy = expected_elements_for_pixels_vector * sizeof(float);
        std::memcpy(this->pixels.data(), data, data_bytes_to_copy);
    } else if (format == vk::Format::eR8G8B8A8Unorm || format == vk::Format::eR8G8B8A8Srgb) {
        const auto* ucData = static_cast<const stbi_uc*>(data);
        for (size_t i = 0; i < expected_elements_for_pixels_vector; ++i) {
            this->pixels[i] = static_cast<float>(ucData[i]) / 255.0f;
        }
    }
    else
        throw std::runtime_error("Unsupported texture format for Texture constructor (raw data): " + std::to_string(static_cast<int>(format)) + ". Name: " + name);

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

glm::vec4 Texture::sample(glm::vec2 uv) const {
    if (pixels.empty() || width == 0 || height == 0)
        return {1.0f, 0.0f, 1.0f, 1.0f}; //Return Pink if texture is empty

    uv = glm::fract(uv);

    int x = static_cast<int>(uv.x * width + EPSILON);
    int y = static_cast<int>(uv.y * height + EPSILON);

    x = std::min(std::max(x, 0), width - 1);
    y = std::min(std::max(y, 0), height - 1);

    size_t index = (static_cast<size_t>(y) * width + x) * 4; 

    if (index + 3 >= pixels.size())
        return glm::vec4(0.0f);

    return {pixels[index], pixels[index + 1], pixels[index + 2], pixels[index + 3]};
}
