#include "Texture.h"

Texture::Texture(const std::shared_ptr<Context> &context, const std::string& filepath) : image(context, filepath)
{
    //Create sampler
    vk::SamplerCreateInfo samplerInfo;
    samplerInfo.setMagFilter(vk::Filter::eLinear);
    samplerInfo.setMinFilter(vk::Filter::eLinear);
    samplerInfo.setAddressModeU(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeV(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAddressModeW(vk::SamplerAddressMode::eRepeat);
    samplerInfo.setAnisotropyEnable(true);
    samplerInfo.setMaxAnisotropy(16.0f);
    sampler = context->device->createSamplerUnique(samplerInfo); //Unique = automatic managed, member so it stays alive

    descriptorInfo.setImageView(*image.view);
    descriptorInfo.setSampler(*sampler);
    descriptorInfo.setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
}