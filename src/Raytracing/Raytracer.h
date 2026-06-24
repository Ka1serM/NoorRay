#pragma once

#include <iostream>

#include "Log.h"
#include "Vulkan/Image.h"
#include "Scene/Scene.h"
#include "../Shaders/Shared.h"

class Raytracer {
protected:
    Context& context;
    Scene& scene;
    uint32_t width, height;
    Image outputColor;
    Image outputAlbedo;
    Image outputNormal;
    Image outputCrypto;
    Image outputPosition;
    Image outputMaterial;

public:
    Raytracer(Scene& scene, const uint32_t width, const uint32_t height)
        : context(scene.getContext()),
          scene(scene),
          width(width),
          height(height),
          outputColor(scene.getContext(), width, height, vk::Format::eR32G32B32A32Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
          outputAlbedo(scene.getContext(), width, height, vk::Format::eR8G8B8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
          outputNormal(scene.getContext(), width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
          outputCrypto(scene.getContext(), width, height, vk::Format::eR32Uint, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
          outputPosition(scene.getContext(), width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst),
          outputMaterial(scene.getContext(), width, height, vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst)
    {}

    virtual ~Raytracer()
    {
        LOG_INFO("Destroying Raytracer");
    }

    Image& getOutputColor()    { return outputColor; }
    Image& getOutputAlbedo()   { return outputAlbedo; }
    Image& getOutputNormal()   { return outputNormal; }
    Image& getOutputCrypto()   { return outputCrypto; }
    Image& getOutputPosition() { return outputPosition; }
    Image& getOutputMaterial() { return outputMaterial; }

    uint32_t getWidth()  const { return width; }
    uint32_t getHeight() const { return height; }

    virtual void resize(const uint32_t newWidth, const uint32_t newHeight) {
        if (newWidth == 0 || newHeight == 0 || (newWidth == width && newHeight == height))
            return;

        width  = newWidth;
        height = newHeight;

        outputColor    = Image(context, width, height, vk::Format::eR32G32B32A32Sfloat,  vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
        outputAlbedo   = Image(context, width, height, vk::Format::eR8G8B8A8Unorm,       vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
        outputNormal   = Image(context, width, height, vk::Format::eR16G16B16A16Sfloat,  vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
        outputCrypto   = Image(context, width, height, vk::Format::eR32Uint,             vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
        outputPosition = Image(context, width, height, vk::Format::eR16G16B16A16Sfloat,  vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
        outputMaterial = Image(context, width, height, vk::Format::eR16G16B16A16Sfloat,  vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);

        LOG_INFO("Raytracer resized to " << width << "x" << height);
    }

    virtual void render(const vk::CommandBuffer& commandBuffer, const PushData& pushConstants) = 0;
    virtual void updateTLAS()     = 0;
    virtual void updateTextures() = 0;
    virtual void updateMeshes()   = 0;
    virtual void updateSceneSettings(const SceneSettings&) {}
};
