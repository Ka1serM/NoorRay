#pragma once

#include <cstdint>
#include <string>

#include <nanobind/ndarray.h>

#include "RenderSession.h"

class PyRenderSession
{
public:
    PyRenderSession(uint32_t width, uint32_t height);

    void importFile(const std::string& path);
    void readScene(const std::string& path);
    void loadScene(const std::string& path);
    void addPerspectiveCamera(
        nanobind::ndarray<const float, nanobind::shape<3>, nanobind::c_contig> position,
        float focalLengthMm);
    void setCameraToWorld(
        nanobind::ndarray<const float, nanobind::shape<4, 4>, nanobind::c_contig> matrix);
    void setCameraFocalLength(float focalLengthMm);
    void setSamples(int samples);
    void setMaxSamples(int samples);
    void setMaxBounces(int bounces);
    void setExposure(float exposure);
    nanobind::ndarray<nanobind::numpy, float, nanobind::shape<-1, -1, 4>> render(uint32_t spp);

    uint32_t width() const;
    uint32_t height() const;
    uint32_t gaussianCount() const;

private:
    noorray::RenderSession session;
};
