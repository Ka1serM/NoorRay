#pragma once

#include <iostream>

#include "Mesh/MeshAsset.h"
#include "Vulkan/Image.h"
#include "Scene/Scene.h"
#include "../Shaders/SharedStructs.h"

class Raytracer {
protected:
    Context& context;
    Scene& scene;
    uint32_t width, height;
    Image outputColor;
    Image outputAlbedo;
    Image outputNormal;
    Image outputCrypto;
    
public:
      Raytracer(Scene& scene, const uint32_t width, const uint32_t height): context(scene.getContext()),
      scene(scene),
      width(width),
      height(height),
      outputColor(scene.getContext(), width, height, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      outputAlbedo(scene.getContext(), width, height, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      outputNormal(scene.getContext(), width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
      outputCrypto(scene.getContext(), width, height, vk::Format::eR32Uint, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst)
    {
    }

    virtual ~Raytracer()
    {
        std::cout << "Destroying Raytracer" << std::endl;
    }
    
    Image& getOutputColor() { return outputColor; }
    Image& getOutputAlbedo() { return outputAlbedo; }
    Image& getOutputNormal() { return outputNormal; }
    Image& getOutputCrypto() { return outputCrypto; }
    
    uint32_t getWidth() const { return width; }
    uint32_t getHeight() const { return height; }

    // Pure virtual since implementations differ
    virtual void render(const vk::CommandBuffer& commandBuffer, const PushConstantsData& pushConstants) = 0;
    virtual void updateTLAS() = 0; 
    virtual void updateTextures() = 0;
    virtual void updateMeshes() = 0;
};

