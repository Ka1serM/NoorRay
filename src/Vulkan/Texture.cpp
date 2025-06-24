#include "Texture.h"
#include <stdexcept>
#include <vector>
#include <stb_image.h>

#include "Utils.h"

Texture::Texture(Context& context, const std::string& hdrFilepath, vk::Format format)
: image([&]() -> Image {
    int texWidth, texHeight, texChannels;
    float* rawPixels = stbi_loadf(hdrFilepath.c_str(), &texWidth, &texHeight, &texChannels, 4); // force 4 channels (RGBA)
    if (!rawPixels)
        throw std::runtime_error(std::string("Failed to load HDR texture from file: ") + hdrFilepath);

    Image img(context, rawPixels, texWidth, texHeight, format);

    stbi_image_free(rawPixels);
    return img;
}())
{
    name = Utils::nameFromPath(hdrFilepath);
    createSampler(context);
}

Texture::Texture(Context& context, const std::string& name, const void* rgbaData, int width, int height)
    : image(context, rgbaData, width, height), name(name)
{
    createSampler(context);
}

Texture::Texture(Context& context, const std::string& filepath)
: image([&]() -> Image {
    int texWidth, texHeight, texChannels;
    stbi_uc* rawPixels = stbi_load(filepath.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    if (!rawPixels)
        throw std::runtime_error(std::string("Failed to load texture from file: ") + filepath);

    Image img(context, rawPixels, texWidth, texHeight);

    stbi_image_free(rawPixels);
    return img;
}())
{
    name = Utils::nameFromPath(filepath);
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