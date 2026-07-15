#include "Camera/Sensor.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "CUDA/rstd/Allocator.h"
#include "CUDA/rstd/Memory.h"
#include "CUDA/Checks.h"
#include "CUDA/ManagedMemory.h"
#include "Log.h"
#include "libross/foundation/gpu/types/Allocator.h"
#include "libross/imaging/imagesensor/ImageSensorReader.h"
#include "portable-file-dialogs.h"

Sensor::~Sensor() = default;

Sensor::Sensor(const Sensor& other)
    : TaggedSensor(nullptr)
    , widthMm(other.width())
    , heightMm(other.height())
    , resolutionWidth(other.resolutionX())
    , resolutionHeight(other.resolutionY())
{
    std::snprintf(imageSensorPath, sizeof(imageSensorPath), "%s",
        std::string(other.getImageSensorPath()).c_str());
    std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%s",
        other.imageSensorLoadStatus);
}

Sensor& Sensor::operator=(const Sensor& other)
{
    if (this == &other)
        return *this;
    widthMm = other.widthMm;
    heightMm = other.heightMm;
    resolutionWidth = other.resolutionWidth;
    resolutionHeight = other.resolutionHeight;
    std::memcpy(imageSensorPath, other.imageSensorPath, sizeof(imageSensorPath));
    std::memcpy(imageSensorLoadStatus, other.imageSensorLoadStatus,
        sizeof(imageSensorLoadStatus));
    imageSensorDialog = nullptr;
    return *this;
}

RectangularSensor::RectangularSensor(const Sensor& other)
    : TaggedBase(other)
{
}

ScatterPsfSensor::ScatterPsfSensor() = default;

ScatterPsfSensor::ScatterPsfSensor(const Sensor& other)
    : TaggedBase(other), psfGridPath(other.getPsfGridPath())
{
    if (!psfGridPath.empty())
        loadPsfGrid();
}

GatherPsfSensor::GatherPsfSensor() = default;

GatherPsfSensor::GatherPsfSensor(const Sensor& other)
    : TaggedBase(other), psfGridPath(other.getPsfGridPath())
{
    if (!psfGridPath.empty())
        loadPsfGrid();
}

namespace {
template <typename PsfSensor>
void freePsfGrid(PsfSensor& sensor)
{
    if (!sensor.psfGrid)
        return;

    nr::synchronizeBeforeManagedMutation("PSF grid free");

    sensor.psfGrid.reset();
}

template <typename PsfSensor>
void loadPsfGrid(PsfSensor& sensor, const char* sensorName)
{
    nr::synchronizeBeforeManagedMutation("PSF grid load");
    freePsfGrid(sensor);

    if (sensor.psfGridPath.empty()) {
        sensor.psfLoadStatus = "PSF grid path is required";
        return;
    }

    try {
        nr::rstd::allocator<ross::InterpolatedPsfGrid> allocator;
        sensor.psfGrid.reset(allocator.allocate(1));
        allocator.construct(sensor.psfGrid.get(), std::filesystem::path(sensor.psfGridPath));
        sensor.psfLoadStatus = "loaded, psf bins: " + std::to_string(sensor.psfGrid->metadata.psfs.size());
        LOG_INFO(sensorName << ": " << sensor.psfLoadStatus);
    } catch (const std::exception& e) {
        freePsfGrid(sensor);
        sensor.psfLoadStatus = e.what();
        LOG_ERROR(sensorName << ": " << sensor.psfLoadStatus);
    }
}

}

std::string_view Sensor::getImageSensorPath() const
{
    return imageSensorPath;
}

void Sensor::setImageSensorPath(std::string_view path)
{
    const size_t copyCount = std::min(path.size(), sizeof(imageSensorPath) - 1);
    std::memcpy(imageSensorPath, path.data(), copyCount);
    imageSensorPath[copyCount] = '\0';
}

bool Sensor::loadImageSensorDimensions()
{
    if (imageSensorPath[0] == '\0') {
        std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%s",
            "Sensor file path is required");
        return false;
    }

    try {
        const ross::ImageSensor loadedSensor = ross::ImageSensorReader::readFile(imageSensorPath);
        nr::synchronizeBeforeManagedMutation("Image sensor dimensions");
        widthMm = std::max(0.001f, loadedSensor.dimensions.width.millimeter());
        heightMm = std::max(0.001f, loadedSensor.dimensions.height.millimeter());
        resolutionWidth = std::max(1u, static_cast<unsigned int>(loadedSensor.resolution.width));
        resolutionHeight = std::max(1u, static_cast<unsigned int>(loadedSensor.resolution.height));
        std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%ux%u",
            loadedSensor.resolution.width, loadedSensor.resolution.height);
        LOG_INFO("Sensor: loaded image sensor " << imageSensorPath);
        return true;
    } catch (const std::exception& e) {
        std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%s", e.what());
        LOG_ERROR("Sensor: " << imageSensorLoadStatus);
        return false;
    }
}

SensorType Sensor::getType() const
{
    if (Is<ScatterPsfSensor>()) return SensorType::ScatterPsf;
    if (Is<GatherPsfSensor>()) return SensorType::GatherPsf;
    return SensorType::Rectangular;
}

std::string Sensor::getPsfGridPath() const
{
    if (const auto* scatter = CastOrNullptr<ScatterPsfSensor>())
        return scatter->psfGridPath;
    if (const auto* gather = CastOrNullptr<GatherPsfSensor>())
        return gather->psfGridPath;
    return {};
}

void Sensor::setPsfGridPath(std::string path)
{
    if (auto* scatter = CastOrNullptr<ScatterPsfSensor>())
    {
        scatter->psfGridPath = std::move(path);
        return;
    }
    if (auto* gather = CastOrNullptr<GatherPsfSensor>())
    {
        gather->psfGridPath = std::move(path);
        return;
    }
    throw std::runtime_error("Select a scatter or gather PSF sensor before loading a PSF grid");
}

void Sensor::loadPsfGrid(std::string path)
{
    setPsfGridPath(std::move(path));
    reloadPsfGrid();
}

uint32_t Sensor::reloadPsfGrid()
{
    if (auto* scatter = CastOrNullptr<ScatterPsfSensor>()) {
        scatter->loadPsfGrid();
        return scatter->psfBinCount();
    }
    if (auto* gather = CastOrNullptr<GatherPsfSensor>()) {
        gather->loadPsfGrid();
        return gather->psfBinCount();
    }
    return 0;
}


ScatterPsfSensor::~ScatterPsfSensor()
{
    freePsfGrid();
}

void ScatterPsfSensor::freePsfGrid()
{
    ::freePsfGrid(*this);
}

void ScatterPsfSensor::loadPsfGrid()
{
    ::loadPsfGrid(*this, "ScatterPsfSensor");
}

uint32_t ScatterPsfSensor::psfBinCount() const
{
    return psfGrid ? static_cast<uint32_t>(psfGrid->metadata.psfs.size()) : 0u;
}


GatherPsfSensor::~GatherPsfSensor()
{
    freePsfGrid();
    freeScratch(nullptr);
}

void GatherPsfSensor::freePsfGrid()
{
    ::freePsfGrid(*this);
}

void GatherPsfSensor::loadPsfGrid()
{
    ::loadPsfGrid(*this, "GatherPsfSensor");
}

uint32_t GatherPsfSensor::psfBinCount() const
{
    return psfGrid ? static_cast<uint32_t>(psfGrid->metadata.psfs.size()) : 0u;
}

void GatherPsfSensor::freeScratch(cudaStream_t stream) noexcept
{
    (void)stream;
    psfGatherBuckets.reset();
    psfGatherBucketCapacity = 0;
}

void GatherPsfSensor::prepareFrame(const uint32_t width, const uint32_t height,
    const bool resetAccumulation, const cudaStream_t stream,
    PsfGatherBucketSample*& buckets, uint32_t& binCount)
{
    buckets = nullptr;
    binCount = psfBinCount();
    if (!psfGrid || binCount == 0)
        return;

    const size_t required = static_cast<size_t>(width) * height * binCount;
    if (required > psfGatherBucketCapacity) {
        freeScratch(stream);
        psfGatherBuckets.allocate(sizeof(PsfGatherBucketSample) * required, stream);
        psfGatherBucketCapacity = required;
    }

    buckets = psfGatherBuckets.as<PsfGatherBucketSample>();
    if (resetAccumulation)
        NR_GPU_CHECK(cudaMemsetAsync(psfGatherBuckets.get(), 0,
            sizeof(PsfGatherBucketSample) * psfGatherBucketCapacity, stream));
}
