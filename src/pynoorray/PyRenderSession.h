#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

#include <cuda_runtime_api.h>
#include <nanobind/ndarray.h>

#include "RenderSession.h"

class PyRenderSession
{
public:
    PyRenderSession(uint32_t width, uint32_t height);
    ~PyRenderSession();

    PyRenderSession(const PyRenderSession&) = delete;
    PyRenderSession& operator=(const PyRenderSession&) = delete;

    void loadScene(const std::string& scenePath);
    void setGaussianOpacity(uint32_t gaussianIndex, float opacity);
    nanobind::ndarray<nanobind::pytorch, float, nanobind::shape<-1, -1, 4>, nanobind::device::cuda>
    render(uint32_t spp);
    uint32_t width() const;
    uint32_t height() const;
    uint32_t gaussianCount() const;

private:
    void ensureOutputBuffer();

    noorray::RenderSession session;
    float* outputDevice{};
    std::size_t outputBytes{};
};
