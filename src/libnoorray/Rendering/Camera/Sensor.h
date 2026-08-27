#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Backend/Host/Platform.h"
#include "Materials/Shading/Spectrum.h"

namespace pfd {
class open_file;
class save_file;
}

class RectangularSensor;
class Sensor;

enum class SensorType : int
{
    Rectangular,
};

enum class SensorOrigin : uint8_t {
    UpperLeft,
    LowerLeft,
};

// Selects how the physical sensor is fitted when the render target has a
// different aspect ratio.
enum class SensorFit : uint8_t {
    Stretch,
    Horizontal,
    Vertical,
};

struct SensorSampleContext {
    glm::vec4* accumulation{};
    uint32_t width{};
    uint32_t height{};
    uint32_t totalAccumulated{};
    float alpha{1.0f};
    uint32_t writeOutput{1};
    const float* cieX{};
    const float* cieY{};
    const float* cieZ{};
};

class Sensor {
public:
    Sensor() = default;
    Sensor(const Sensor& other);
    Sensor& operator=(const Sensor& other);
    virtual ~Sensor();
    Sensor* ptr() { return this; }
    const Sensor* ptr() const { return this; }
    explicit operator bool() const { return true; }
    template <typename F> decltype(auto) DispatchCPU(F&& f) { return f(this); }
    template <typename F> decltype(auto) DispatchCPU(F&& f) const { return f(this); }

    float widthMm{5.784f};
    float heightMm{3.264f};
    // The physical dimensions above remain authoritative for image-sensor
    // and optical calculations. These are the dimensions sampled by camera
    // rays after applying the host camera's sensor-fit policy.
    float filmWidthMm{5.784f};
    float filmHeightMm{3.264f};
    uint32_t resolutionWidth{1280};
    uint32_t resolutionHeight{720};
    SensorOrigin sensorOrigin{SensorOrigin::UpperLeft};

    char imageSensorPath[512]{};
    char imageSensorLoadStatus[512]{};
    pfd::open_file* imageSensorDialog{};

    NR_CPU_GPU float width() const;
    NR_CPU_GPU float height() const;
    NR_CPU_GPU float filmWidth() const;
    NR_CPU_GPU float filmHeight() const;
    NR_CPU_GPU uint32_t resolutionX() const;
    NR_CPU_GPU uint32_t resolutionY() const;
    NR_CPU_GPU glm::uvec2 resolution() const;
    NR_CPU_GPU SensorOrigin origin() const;
    NR_CPU_GPU void setResolution(uint32_t w, uint32_t h);
    NR_CPU_GPU void setDimensionsMm(float w, float h);
    NR_CPU_GPU void setFilmDimensionsMm(float w, float h);
    NR_CPU_GPU void setFilmFit(SensorFit fit, uint32_t renderWidth,
        uint32_t renderHeight);
    NR_CPU_GPU void setOrigin(SensorOrigin value);
    NR_CPU_GPU void copyPhysicalFrom(const Sensor& other);
    NR_CPU_GPU float aspectRatio() const;

    std::string_view getImageSensorPath() const;
    void setImageSensorPath(std::string_view path);
    bool loadImageSensorDimensions();
    SensorType getType() const;
    void requestType(SensorType type) { requestedType = static_cast<int>(type); }
    bool consumeRequestedType(SensorType& type)
    {
        if (requestedType < 0)
            return false;
        type = static_cast<SensorType>(requestedType);
        requestedType = -1;
        return true;
    }

    bool renderUi();

private:
    int requestedType{-1};
};

NR_CPU_GPU inline glm::vec3 sensorRGBFromSpectrum(
    const SampledSpectrum& L, const SampledWavelengths& wl,
    const float* cieX, const float* cieY, const float* cieZ)
{
    return xyzToLinearSRGB(spectrumToXYZ(L, wl, cieX, cieY, cieZ));
}

NR_CPU_GPU inline void sensorAtomicAdd(double* ptr, double value)
{
    *ptr += value;
}

NR_CPU_GPU inline void sensorAtomicAdd(float* ptr, float value)
{
    *ptr += value;
}

#include "Rendering/Camera/RectangularSensor.h"

NR_CPU_GPU inline float Sensor::width() const
{
    return widthMm;
}

NR_CPU_GPU inline float Sensor::height() const
{
    return heightMm;
}

NR_CPU_GPU inline float Sensor::filmWidth() const
{
    return filmWidthMm;
}

NR_CPU_GPU inline float Sensor::filmHeight() const
{
    return filmHeightMm;
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

NR_CPU_GPU inline SensorOrigin Sensor::origin() const
{
    return sensorOrigin;
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
    filmWidthMm = widthMm;
    filmHeightMm = heightMm;
}

NR_CPU_GPU inline void Sensor::setFilmDimensionsMm(float w, float h)
{
    filmWidthMm = std::max(0.001f, w);
    filmHeightMm = std::max(0.001f, h);
}

NR_CPU_GPU inline void Sensor::setFilmFit(
    const SensorFit fit, const uint32_t renderWidth,
    const uint32_t renderHeight)
{
    if (renderWidth == 0 || renderHeight == 0) {
        setFilmDimensionsMm(width(), height());
        return;
    }

    const float renderAspect = static_cast<float>(renderWidth)
        / static_cast<float>(renderHeight);
    if (fit == SensorFit::Stretch) {
        setFilmDimensionsMm(width(), height());
    } else if (fit == SensorFit::Vertical) {
        setFilmDimensionsMm(height() * renderAspect, height());
    } else {
        setFilmDimensionsMm(width(), width() / renderAspect);
    }
}

NR_CPU_GPU inline void Sensor::setOrigin(const SensorOrigin value)
{
    sensorOrigin = value;
}

NR_CPU_GPU inline void Sensor::copyPhysicalFrom(const Sensor& other)
{
    setDimensionsMm(other.width(), other.height());
    setResolution(other.resolutionX(), other.resolutionY());
    setOrigin(other.origin());
}

NR_CPU_GPU inline float Sensor::aspectRatio() const
{
    return width() / height();
}
