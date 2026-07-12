#include "PyRenderSession.h"

#include <algorithm>
#include <cstring>

#include <nanobind/nanobind.h>

namespace nb = nanobind;

namespace
{
glm::mat4 rowMajorMatrix(const float* values)
{
    glm::mat4 result;
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            result[column][row] = values[row * 4 + column];
    return result;
}
}

PyRenderSession::PyRenderSession(const uint32_t width, const uint32_t height) : session(width, height) {}

void PyRenderSession::importFile(const std::string& path) { session.importFile(path); }
void PyRenderSession::readScene(const std::string& path) { session.readScene(path); }
void PyRenderSession::loadScene(const std::string& path) { session.loadScene(path); }

void PyRenderSession::addPerspectiveCamera(
    nb::ndarray<const float, nb::shape<3>, nb::c_contig> position, const float focalLengthMm)
{
    session.addPerspectiveCamera(glm::vec3(position(0), position(1), position(2)), focalLengthMm);
}

void PyRenderSession::setCameraToWorld(
    nb::ndarray<const float, nb::shape<4, 4>, nb::c_contig> matrix)
{
    session.setCameraToWorld(rowMajorMatrix(matrix.data()));
}

void PyRenderSession::setCameraFocalLength(const float focalLengthMm) { session.setCameraFocalLength(focalLengthMm); }
void PyRenderSession::setSamples(const int samples) { session.setSamples(samples); }
void PyRenderSession::setMaxSamples(const int samples) { session.setMaxSamples(samples); }
void PyRenderSession::setMaxBounces(const int bounces) { session.setMaxBounces(bounces); }
void PyRenderSession::setExposure(const float exposure) { session.setExposure(exposure); }

nb::ndarray<nb::numpy, float, nb::shape<-1, -1, 4>> PyRenderSession::render(const uint32_t spp)
{
    const Bitmap bitmap = session.renderBitmap(std::max(1u, spp));
    const size_t valueCount = static_cast<size_t>(bitmap.width()) * bitmap.height() * 4;
    auto* pixels = new float[valueCount];
    std::memcpy(pixels, bitmap.rgba(), valueCount * sizeof(float));
    nb::capsule owner(pixels, [](void* pointer) noexcept { delete[] static_cast<float*>(pointer); });
    const size_t shape[3] = {bitmap.height(), bitmap.width(), 4};
    return {pixels, 3, shape, owner};
}

uint32_t PyRenderSession::width() const { return session.width(); }
uint32_t PyRenderSession::height() const { return session.height(); }
uint32_t PyRenderSession::gaussianCount() const { return session.gaussianCount(); }
