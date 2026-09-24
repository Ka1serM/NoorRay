#include "Materials/Material.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

void Material::upload(noorrhi::Device& device,
    const std::function<std::uint32_t(std::uint32_t)>& resolveTexture)
{
    std::vector<std::uint32_t> words = shaderProgram.parameters;
    for (const MaterialTextureWord& texture : shaderProgram.textures)
        words[texture.word] = resolveTexture(texture.texture);
    shaderParameters = device.buffer<std::uint32_t>(std::max<std::size_t>(words.size(), 1u));
    if (!words.empty())
        shaderParameters.upload(std::span<const std::uint32_t>(words));

    if (!*this)
        allocate(device);
    data.shaderParameters = shaderParameters.ptr().address;
    commit();
}
