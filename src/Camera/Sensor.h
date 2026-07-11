#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>
#ifndef NR_GPU_CODE
#include <string>
#endif

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "CUDA/Annotations.h"
#include "CUDA/TaggedPointer.h"
#include "Raytracing/Spectrum.h"

namespace ross {
class InterpolatedPsfGrid;
}

namespace pfd {
class open_file;
}

class RectangularSensor;
class ScatterPsfSensor;
class GatherPsfSensor;

struct PsfGatherBucketSample {
    double rgbSum[3]{};
    double count{};
};

struct SensorSampleContext {
    glm::vec4* accumulation{};
    PsfGatherBucketSample* psfBuckets{};
    uint32_t width{};
    uint32_t height{};
    uint32_t totalAccumulated{};
    float alpha{1.0f};
    const float* cieX{};
    const float* cieY{};
    const float* cieZ{};
};

using TaggedSensor = nr::TaggedPointer<RectangularSensor, ScatterPsfSensor, GatherPsfSensor>;

class Sensor : public TaggedSensor {
public:
    using TaggedSensor::TaggedSensor;

    float widthMm{5.784f};
    float heightMm{3.264f};
    uint32_t resolutionWidth{1280};
    uint32_t resolutionHeight{720};

    char imageSensorPath[512]{};
    char imageSensorLoadStatus[512]{};
    pfd::open_file* imageSensorDialog{};

    NR_CPU_GPU float width() const;
    NR_CPU_GPU float height() const;
    NR_CPU_GPU uint32_t resolutionX() const;
    NR_CPU_GPU uint32_t resolutionY() const;
    NR_CPU_GPU glm::uvec2 resolution() const;
    NR_CPU_GPU void setResolution(uint32_t w, uint32_t h);
    NR_CPU_GPU void setDimensionsMm(float w, float h);
    NR_CPU_GPU void copyPhysicalFrom(const Sensor& other);
    NR_CPU_GPU float aspectRatio() const;

#ifndef NR_GPU_CODE
    std::string_view getImageSensorPath() const;
    void setImageSensorPath(std::string_view path);
    bool loadImageSensorDimensions();
    void freeConcrete();
    void moveConcreteFrom(Sensor& other);
    void cloneConcreteFrom(const Sensor& other);
    void allocateRectangular();
    void allocateScatterPsf();
    void allocateGatherPsf();
    uint32_t reloadPsfGrid();
#endif

    bool renderUi();
};

NR_CPU_GPU inline glm::vec3 sensorRGBFromSpectrum(
    const SampledSpectrum& L, const SampledWavelengths& wl,
    const float* cieX, const float* cieY, const float* cieZ)
{
    return xyzToLinearSRGB(spectrumToXYZ(L, wl, cieX, cieY, cieZ));
}

NR_CPU_GPU inline void sensorAtomicAdd(double* ptr, double value)
{
#if defined(__CUDA_ARCH__)
    atomicAdd(ptr, value);
#else
    *ptr += value;
#endif
}

NR_CPU_GPU inline void sensorAtomicAdd(float* ptr, float value)
{
#if defined(__CUDA_ARCH__)
    atomicAdd(ptr, value);
#else
    *ptr += value;
#endif
}

#include "Camera/RectangularSensor.h"
#include "Camera/ScatterPsfSensor.h"
#include "Camera/GatherPsfSensor.h"

NR_CPU_GPU inline float Sensor::width() const
{
    return widthMm;
}

NR_CPU_GPU inline float Sensor::height() const
{
    return heightMm;
}

NR_CPU_GPU inline uint32_t Sensor::resolutionX() const
{
    return resolutionWidth;
}

NR_CPU_GPU inline uint32_t Sensor::resolutionY() const
{
    return resolutionHeight;
}

NR_CPU_GPU inline glm::uvec2 Sensor::resolution() const
{
    return {resolutionX(), resolutionY()};
}

NR_CPU_GPU inline void Sensor::setResolution(uint32_t w, uint32_t h)
{
    resolutionWidth = w;
    resolutionHeight = h;
}

NR_CPU_GPU inline void Sensor::setDimensionsMm(float w, float h)
{
    widthMm = std::max(0.001f, w);
    heightMm = std::max(0.001f, h);
}

NR_CPU_GPU inline void Sensor::copyPhysicalFrom(const Sensor& other)
{
    setDimensionsMm(other.width(), other.height());
    setResolution(other.resolutionX(), other.resolutionY());
}

NR_CPU_GPU inline float Sensor::aspectRatio() const
{
    return width() / height();
}
