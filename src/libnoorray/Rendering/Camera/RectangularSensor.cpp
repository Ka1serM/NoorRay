#include "Rendering/Camera/Sensor.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <stdexcept>
#include "Backend/Host/MutationBarrier.h"
#include "Log.h"

Sensor::~Sensor() = default;
Sensor::Sensor(const Sensor& other) : widthMm(other.width()), heightMm(other.height()), filmWidthMm(other.filmWidth()), filmHeightMm(other.filmHeight()), resolutionWidth(other.resolutionX()), resolutionHeight(other.resolutionY()), sensorOrigin(other.origin()) { std::memcpy(imageSensorPath, other.imageSensorPath, sizeof(imageSensorPath)); std::memcpy(imageSensorLoadStatus, other.imageSensorLoadStatus, sizeof(imageSensorLoadStatus)); }
Sensor& Sensor::operator=(const Sensor& other) { if (this == &other) return *this; widthMm=other.widthMm; heightMm=other.heightMm; filmWidthMm=other.filmWidthMm; filmHeightMm=other.filmHeightMm; resolutionWidth=other.resolutionWidth; resolutionHeight=other.resolutionHeight; sensorOrigin=other.sensorOrigin; std::memcpy(imageSensorPath,other.imageSensorPath,sizeof(imageSensorPath)); std::memcpy(imageSensorLoadStatus,other.imageSensorLoadStatus,sizeof(imageSensorLoadStatus)); imageSensorDialog=nullptr; return *this; }
RectangularSensor::RectangularSensor(const Sensor& other) : Sensor(other) {}
std::string_view Sensor::getImageSensorPath() const { return imageSensorPath; }
void Sensor::setImageSensorPath(std::string_view path) { const size_t n=std::min(path.size(),sizeof(imageSensorPath)-1); std::memcpy(imageSensorPath,path.data(),n); imageSensorPath[n]='\0'; }
bool Sensor::loadImageSensorDimensions() {
    if (!imageSensorPath[0]) { std::snprintf(imageSensorLoadStatus,sizeof(imageSensorLoadStatus),"Sensor file path is required"); return false; }
    try {
        std::ifstream file(imageSensorPath);
        if (!file) throw std::runtime_error("Cannot open sensor file");
        const std::string text((std::istreambuf_iterator<char>(file)), {});
        const auto number = [&](std::initializer_list<std::string> patterns, const char* field) {
            std::smatch match;
            for (const auto& expression : patterns) {
                if (std::regex_search(text, match, std::regex(expression))) return std::stof(match[1]);
            }
            throw std::runtime_error(std::string("Missing sensor field '") + field + "'");
        };
        // Accept the nested camera-sensor schema used by the catalog assets as
        // well as the legacy flat schema written by older NoorRay scenes.
        const float width = number({
            R"("width_mm"\s*:\s*([-+0-9.eE]+))",
            R"("width"\s*:\s*([-+0-9.eE]+))"}, "width");
        const float height = number({
            R"("height_mm"\s*:\s*([-+0-9.eE]+))",
            R"("height"\s*:\s*([-+0-9.eE]+))"}, "height");
        const float resolutionX = number({
            R"("resolutionX"\s*:\s*([-+0-9.eE]+))",
            R"("resolution"\s*:\s*\{[^}]*"width"\s*:\s*([-+0-9.eE]+))"}, "resolution width");
        const float resolutionY = number({
            R"("resolutionY"\s*:\s*([-+0-9.eE]+))",
            R"("resolution"\s*:\s*\{[^}]*"height"\s*:\s*([-+0-9.eE]+))"}, "resolution height");
        nr::synchronizeBeforeManagedMutation("Image sensor dimensions");
        setDimensionsMm(width, height);
        resolutionWidth = std::max(1u, static_cast<uint32_t>(resolutionX));
        resolutionHeight = std::max(1u, static_cast<uint32_t>(resolutionY));
        std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%ux%u", resolutionWidth, resolutionHeight);
        return true;
    }
    catch(const std::exception& e) { std::snprintf(imageSensorLoadStatus,sizeof(imageSensorLoadStatus),"%s",e.what()); LOG_ERROR("Sensor: " << imageSensorLoadStatus); return false; }
}
SensorType Sensor::getType() const { return SensorType::Rectangular; }
