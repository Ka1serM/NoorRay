#include "PyRenderSession.h"

#include <stdexcept>

#include <nanobind/nanobind.h>

#include "CUDA/Checks.h"

namespace nb = nanobind;

PyRenderSession::PyRenderSession(const uint32_t width, const uint32_t height)
    : session(width, height)
{
}

PyRenderSession::~PyRenderSession()
{
    if (outputDevice)
        cudaFree(outputDevice);
}

void PyRenderSession::loadScene(const std::string& scenePath)
{
    session.loadScene(scenePath);
}

void PyRenderSession::setGaussianOpacity(const uint32_t gaussianIndex, const float opacity)
{
    session.setGaussianOpacity(gaussianIndex, opacity);
}

void PyRenderSession::ensureOutputBuffer()
{
    const std::size_t requiredBytes =
        static_cast<std::size_t>(width()) * height() * 4 * sizeof(float);
    if (requiredBytes == 0)
        throw std::runtime_error("Render output has zero size");
    if (requiredBytes == outputBytes)
        return;

    if (outputDevice)
    {
        cudaFree(outputDevice);
        outputDevice = nullptr;
        outputBytes = 0;
    }

    NR_GPU_CHECK(cudaMalloc(reinterpret_cast<void**>(&outputDevice), requiredBytes));
    outputBytes = requiredBytes;
}

nb::ndarray<nb::pytorch, float, nb::shape<-1, -1, 4>, nb::device::cuda>
PyRenderSession::render(const uint32_t spp)
{
    ensureOutputBuffer();
    session.renderToDevice(outputDevice, spp);
    NR_GPU_CHECK(cudaDeviceSynchronize());

    const std::size_t shape[3] = {height(), width(), 4};
    const nb::object owner = nb::cast(this, nb::rv_policy::reference);
    return nb::ndarray<nb::pytorch, float, nb::shape<-1, -1, 4>, nb::device::cuda>(
        outputDevice,
        3,
        shape,
        owner,
        nullptr,
        nb::dtype<float>(),
        nb::device::cuda::value,
        0);
}

uint32_t PyRenderSession::width() const
{
    return session.width();
}

uint32_t PyRenderSession::height() const
{
    return session.height();
}

uint32_t PyRenderSession::gaussianCount() const
{
    return session.gaussianCount();
}
