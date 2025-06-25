#include "Texture.h"
#include <stdexcept>
#include <vector>
#include <stb_image.h>

#include "Utils.h"

Texture::Texture(Context& context, const std::string& filepath)
    : image([&]() -> Image {
        int texWidth, texHeight, texChannels;

        bool isHdr = stbi_is_hdr(filepath.c_str());
        void* pixelData = nullptr;
        vk::Format format;

        if (isHdr) {
            float* rawPixels = stbi_loadf(filepath.c_str(), &texWidth, &texHeight, &texChannels, 4); // force 4 channels
            if (!rawPixels)
                throw std::runtime_error("Failed to load HDR texture: " + filepath);
            pixelData = rawPixels;
            format = vk::Format::eR32G32B32A32Sfloat;
        } else {
            stbi_uc* rawPixels = stbi_load(filepath.c_str(), &texWidth, &texHeight, &texChannels, 4); // force 4 channels
            if (!rawPixels)
                throw std::runtime_error("Failed to load 8-bit texture: " + filepath);
            pixelData = rawPixels;
            format = vk::Format::eR8G8B8A8Unorm;
        }

        Image img(context, pixelData, texWidth, texHeight, format);

        // Free image data
        stbi_image_free(pixelData);

        return img;
    }())
{
    name = Utils::nameFromPath(filepath);
    createSampler(context);
}

Texture::Texture(Context& context, const std::string& name, const void* data, int width, int height, vk::Format format)
    : image(context, data, width, height, format), name(name)
{
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
    samplerInfo.setAnisotropyEnable(true);
    samplerInfo.setMaxAnisotropy(16.0f);

    sampler = context.device->createSamplerUnique(samplerInfo);

    descriptorInfo.setImageView(*image.view);
    descriptorInfo.setSampler(*sampler);
    descriptorInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}