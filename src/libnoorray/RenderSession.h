#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "IO/Bitmap.h"
#include "Scene/Scene.h"
#include "Vulkan/Context.h"

class Raytracer;

namespace noorray
{

class RenderSession
{
public:
    RenderSession(uint32_t width, uint32_t height);
    ~RenderSession();

    RenderSession(const RenderSession&) = delete;
    RenderSession& operator=(const RenderSession&) = delete;

    void loadScene(const std::string& scenePath);
    Bitmap renderBitmap(uint32_t spp);
    void renderToDevice(float* rgbaDevice, uint32_t spp);
    void setGaussianOpacity(uint32_t gaussianIndex, float opacity);

    uint32_t width() const;
    uint32_t height() const;
    uint32_t gaussianCount() const;

private:
    Context context;
    Scene scene;
    std::unique_ptr<Raytracer> raytracer;
    uint32_t requestedWidth{};
    uint32_t requestedHeight{};
};

}
