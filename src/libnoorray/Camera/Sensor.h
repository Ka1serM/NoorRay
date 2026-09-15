#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

#include <glm/vec2.hpp>

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

    float width() const;
    float height() const;
    float filmWidth() const;
    float filmHeight() const;
    uint32_t resolutionX() const;
    uint32_t resolutionY() const;
    glm::uvec2 resolution() const;
    SensorOrigin origin() const;
    void setResolution(uint32_t w, uint32_t h);
    void setDimensionsMm(float w, float h);
    void setFilmDimensionsMm(float w, float h);
    void setFilmFit(SensorFit fit, uint32_t renderWidth,
        uint32_t renderHeight);
    void setOrigin(SensorOrigin value);
    void copyPhysicalFrom(const Sensor& other);
    float aspectRatio() const;

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

#include "Camera/RectangularSensor.h"

inline float Sensor::width() const
{
    return widthMm;
}

inline float Sensor::height() const
{
    return heightMm;
}

inline float Sensor::filmWidth() const
{
    return filmWidthMm;
}

inline float Sensor::filmHeight() const
{
    return filmHeightMm;
}

inline uint32_t Sensor::resolutionX() const
{
    return resolutionWidth;
}

inline uint32_t Sensor::resolutionY() const
{
    return resolutionHeight;
}

inline glm::uvec2 Sensor::resolution() const
{
    return {resolutionX(), resolutionY()};
}

inline SensorOrigin Sensor::origin() const
{
    return sensorOrigin;
}

inline void Sensor::setResolution(uint32_t w, uint32_t h)
{
    resolutionWidth = w;
    resolutionHeight = h;
}

inline void Sensor::setDimensionsMm(float w, float h)
{
    widthMm = std::max(0.001f, w);
    heightMm = std::max(0.001f, h);
    filmWidthMm = widthMm;
    filmHeightMm = heightMm;
}

inline void Sensor::setFilmDimensionsMm(float w, float h)
{
    filmWidthMm = std::max(0.001f, w);
    filmHeightMm = std::max(0.001f, h);
}

inline void Sensor::setFilmFit(
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

inline void Sensor::setOrigin(const SensorOrigin value)
{
    sensorOrigin = value;
}

inline void Sensor::copyPhysicalFrom(const Sensor& other)
{
    setDimensionsMm(other.width(), other.height());
    setResolution(other.resolutionX(), other.resolutionY());
    setOrigin(other.origin());
}

inline float Sensor::aspectRatio() const
{
    return width() / height();
}
