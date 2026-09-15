#include "Materials/Material.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

void Material::upload(noorrhi::Device& device,
    const std::function<std::uint32_t(std::uint32_t)>& resolveTexture)
{
    if (hasProgram()) {
        bytecode = device.buffer<std::uint32_t>(program.bytecode.size());
        bytecode.upload(std::span<const std::uint32_t>(program.bytecode));

        std::vector<std::uint32_t> handles(program.textureIndices.size());
        for (std::size_t slot = 0; slot < handles.size(); ++slot)
            handles[slot] = resolveTexture(program.textureIndices[slot]);
        textureHandles = device.buffer<std::uint32_t>(
            std::max<std::size_t>(handles.size(), 1u));
        if (!handles.empty())
            textureHandles.upload(std::span<const std::uint32_t>(handles));
    } else if (!textureHandles) {
        textureHandles = device.buffer<std::uint32_t>(1);
    }

    if (!*this)
        allocate(device);
    data.bytecode = bytecode ? bytecode.ptr().address : 0;
    data.textures = textureHandles.ptr().address;
    data.bytecodeLength = static_cast<std::uint32_t>(program.bytecode.size());
    data.stackSize = program.stackSize;
    data.shadowOpaque = shadowOpaque;
    commit();
}
