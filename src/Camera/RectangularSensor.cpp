#include "Camera/Sensor.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <type_traits>
#include <vector>
#include <imgui.h>

#include "CUDA/rstd/Allocator.h"
#include "CUDA/rstd/Memory.h"
#include "CUDA/Checks.h"
#include "Log.h"
#include "UI/ImGuiManager.h"
#include "libross/foundation/gpu/types/Allocator.h"
#include "libross/imaging/imagesensor/ImageSensorReader.h"
#include "portable-file-dialogs.h"

#if defined(NR_CUDA_ACTIVE)
#include <cuda_runtime.h>
#endif

namespace {
constexpr int RectangularSensorTypeIndex = 0;
constexpr int ScatterPsfSensorTypeIndex = 1;
constexpr int GatherPsfSensorTypeIndex = 2;

constexpr std::array<const char*, 3> SensorTypeNames{
    "Rectangular", "Scatter PSF", "Gather PSF"};

void synchronizeBeforeManagedSensorMutation(const char* reason)
{
#if defined(NR_CUDA_ACTIVE)
    const cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess)
        LOG_ERROR(reason << ": CUDA synchronize failed before managed sensor mutation: "
                         << cudaGetErrorString(result));
#else
    (void)reason;
#endif
}

template <typename SensorType>
void allocateSensor(Sensor& sensor)
{
    synchronizeBeforeManagedSensorMutation("Sensor allocation");

    const float previousWidth = sensor.width();
    const float previousHeight = sensor.height();
    const uint32_t previousResolutionX = sensor.resolutionX();
    const uint32_t previousResolutionY = sensor.resolutionY();
    const std::string previousImageSensorPath(sensor.getImageSensorPath());
    std::string previousImageSensorLoadStatus;
    if (sensor.ptr()) {
        previousImageSensorLoadStatus = sensor.DispatchCPU([](const auto* concreteSensor) {
            return std::string(concreteSensor->imageSensorLoadStatus);
        });
    } else {
        previousImageSensorLoadStatus = sensor.imageSensorLoadStatus;
    }
    std::string previousPsfGridPath;
    if (const auto* scatter = sensor.CastOrNullptr<ScatterPsfSensor>())
        previousPsfGridPath = scatter->psfGridPath;
    if (const auto* gather = sensor.CastOrNullptr<GatherPsfSensor>())
        previousPsfGridPath = gather->psfGridPath;

    sensor.freeConcrete();

    nr::rstd::allocator<SensorType> allocator;
    SensorType* concrete = allocator.allocate(1);
    allocator.construct(concrete);
    concrete->setImageSensorPath(previousImageSensorPath);
    std::snprintf(concrete->imageSensorLoadStatus, sizeof(concrete->imageSensorLoadStatus), "%s",
        previousImageSensorLoadStatus.c_str());
    concrete->setDimensionsMm(previousWidth, previousHeight);
    concrete->setResolution(previousResolutionX, previousResolutionY);
    if constexpr (requires { concrete->psfGrid; }) {
        concrete->psfGridPath = previousPsfGridPath;
        if (!concrete->psfGridPath.empty())
            concrete->loadPsfGrid();
    }
    sensor = Sensor(concrete);
}

template <typename SensorType>
void cloneSensor(Sensor& destination, const Sensor& source)
{
    destination.freeConcrete();

    const float width = source.width();
    const float height = source.height();
    const uint32_t resolutionX = source.resolutionX();
    const uint32_t resolutionY = source.resolutionY();
    const std::string imageSensorPath(source.getImageSensorPath());
    std::string imageSensorLoadStatus;
    if (source.ptr()) {
        imageSensorLoadStatus = source.DispatchCPU([](const auto* concreteSensor) {
            return std::string(concreteSensor->imageSensorLoadStatus);
        });
    } else {
        imageSensorLoadStatus = source.imageSensorLoadStatus;
    }

    nr::rstd::allocator<SensorType> allocator;
    SensorType* concrete = allocator.allocate(1);
    allocator.construct(concrete);
    concrete->setImageSensorPath(imageSensorPath);
    std::snprintf(concrete->imageSensorLoadStatus, sizeof(concrete->imageSensorLoadStatus), "%s",
        imageSensorLoadStatus.c_str());
    concrete->setDimensionsMm(width, height);
    concrete->setResolution(resolutionX, resolutionY);

    if constexpr (std::is_same_v<SensorType, ScatterPsfSensor>) {
        if (const auto* scatter = source.CastOrNullptr<ScatterPsfSensor>()) {
            concrete->psfGridPath = scatter->psfGridPath;
            if (!concrete->psfGridPath.empty())
                concrete->loadPsfGrid();
        }
    } else if constexpr (std::is_same_v<SensorType, GatherPsfSensor>) {
        if (const auto* gather = source.CastOrNullptr<GatherPsfSensor>()) {
            concrete->psfGridPath = gather->psfGridPath;
            if (!concrete->psfGridPath.empty())
                concrete->loadPsfGrid();
        }
    }

    destination = Sensor(concrete);
}

bool beginSensorUi(const char* label)
{
    ImGuiManager::tableRowLabel("Sensor");
    return ImGui::TreeNodeEx(label, ImGuiTreeNodeFlags_Framed);
}

bool renderSensorTypeCombo(Sensor& owner, int currentTypeIndex)
{
    ImGuiManager::tableRowLabel("Type");
    int typeIndex = currentTypeIndex;
    if (!ImGui::Combo("##SensorType", &typeIndex, SensorTypeNames.data(), static_cast<int>(SensorTypeNames.size())))
        return false;

    if (typeIndex == currentTypeIndex)
        return false;
    if (typeIndex == RectangularSensorTypeIndex)
        owner.allocateRectangular();
    else if (typeIndex == ScatterPsfSensorTypeIndex)
        owner.allocateScatterPsf();
    else if (typeIndex == GatherPsfSensorTypeIndex)
        owner.allocateGatherPsf();
    return true;
}

bool renderPhysicalSensorRows(Sensor& sensor)
{
    if (sensor.imageSensorDialog && sensor.imageSensorDialog->ready(0)) {
        const auto selection = sensor.imageSensorDialog->result();
        if (!selection.empty()) {
            sensor.setImageSensorPath(selection.front());
            sensor.loadImageSensorDimensions();
        }
        delete sensor.imageSensorDialog;
        sensor.imageSensorDialog = nullptr;
    }

    bool changed = false;

    std::array<char, 512> sensorBuffer{};
    std::snprintf(sensorBuffer.data(), sensorBuffer.size(), "%s", sensor.imageSensorPath);
    const float selectButtonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;

    ImGuiManager::tableRowLabel("Sensor File");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - selectButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##ImageSensorPath", sensorBuffer.data(), sensorBuffer.size())) {
        sensor.setImageSensorPath(sensorBuffer.data());
        changed = true;
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    ImGui::BeginDisabled(sensor.imageSensorDialog != nullptr);
    if (ImGui::Button("Select##ImageSensor", ImVec2(selectButtonWidth, 0))) {
        sensor.imageSensorDialog = new pfd::open_file(
            "Select Sensor File", ".",
            std::vector<std::string>{"Sensor Files", "*.json", "All Files", "*"});
    }
    ImGui::EndDisabled();

    if (ImGui::Button("Load Sensor##ImageSensor")) {
        changed |= sensor.loadImageSensorDimensions();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(sensor.imageSensorLoadStatus);

    float currentWidthMm = sensor.width();
    float currentHeightMm = sensor.height();
    ImGuiManager::dragFloatRow("Width (mm)", currentWidthMm, 0.1f, 0.1f, 500.f, [&](float v) {
        sensor.setDimensionsMm(std::max(v, 0.1f), sensor.height());
        changed = true;
    });
    ImGuiManager::dragFloatRow("Height (mm)", currentHeightMm, 0.1f, 0.1f, 500.f, [&](float v) {
        sensor.setDimensionsMm(sensor.width(), std::max(v, 0.1f));
        changed = true;
    });

    ImGuiManager::tableRowLabel("Resolution");
    int w = static_cast<int>(sensor.resolutionX());
    int h = static_cast<int>(sensor.resolutionY());
    ImGui::PushItemWidth((ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize("x").x
                          - ImGui::GetStyle().ItemSpacing.x * 2.f) * 0.5f);
    if (ImGui::InputInt("##ResW", &w, 0, 0, ImGuiInputTextFlags_CharsDecimal))
        changed = true;
    ImGui::SameLine();
    ImGui::TextUnformatted("x");
    ImGui::SameLine();
    if (ImGui::InputInt("##ResH", &h, 0, 0, ImGuiInputTextFlags_CharsDecimal))
        changed = true;
    ImGui::PopItemWidth();
    sensor.setResolution(
        static_cast<uint32_t>(std::clamp(w, 1, 16384)),
        static_cast<uint32_t>(std::clamp(h, 1, 16384)));

    return changed;
}

template <typename PsfSensor>
void freePsfGrid(PsfSensor& sensor)
{
    if (sensor.psfGrid == nullptr)
        return;

    synchronizeBeforeManagedSensorMutation("PSF grid free");

    ross::rstd::allocator<ross::InterpolatedPsfGrid> allocator;
    allocator.destroy(sensor.psfGrid);
    allocator.deallocate(sensor.psfGrid, 1);
    sensor.psfGrid = nullptr;
}

template <typename PsfSensor>
void loadPsfGrid(PsfSensor& sensor, const char* sensorName)
{
    freePsfGrid(sensor);

    if (sensor.psfGridPath.empty()) {
        sensor.psfLoadStatus = "PSF grid path is required";
        return;
    }

    try {
        ross::rstd::allocator<ross::InterpolatedPsfGrid> allocator;
        sensor.psfGrid = allocator.allocate(1);
        allocator.construct(sensor.psfGrid, std::filesystem::path(sensor.psfGridPath));
        sensor.psfLoadStatus = "loaded, psf bins: " + std::to_string(sensor.psfGrid->metadata.psfs.size());
        LOG_INFO(sensorName << ": " << sensor.psfLoadStatus);
    } catch (const std::exception& e) {
        freePsfGrid(sensor);
        sensor.psfLoadStatus = e.what();
        LOG_ERROR(sensorName << ": " << sensor.psfLoadStatus);
    }
}

template <typename PsfSensor>
bool renderPsfGridRows(PsfSensor& sensor)
{
    if (sensor.psfGridDialog && sensor.psfGridDialog->ready(0)) {
        const auto selection = sensor.psfGridDialog->result();
        if (!selection.empty()) {
            sensor.psfGridPath = selection.front();
            sensor.loadPsfGrid();
        }
        sensor.psfGridDialog.reset();
    }

    bool changed = false;
    std::array<char, 512> psfBuffer{};
    std::snprintf(psfBuffer.data(), psfBuffer.size(), "%s", sensor.psfGridPath.c_str());

    const float selectButtonWidth = ImGui::CalcTextSize("Select").x + ImGui::GetStyle().FramePadding.x * 2.0f;
    ImGuiManager::tableRowLabel("PSF Grid");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - selectButtonWidth - ImGui::GetStyle().ItemSpacing.x);
    if (ImGui::InputText("##SensorPsfGrid", psfBuffer.data(), psfBuffer.size())) {
        sensor.psfGridPath = psfBuffer.data();
        changed = true;
    }
    ImGui::PopItemWidth();
    ImGui::SameLine();
    if (ImGui::Button("Select##SensorPsfGrid")) {
        sensor.psfGridDialog = std::make_unique<pfd::open_file>(
            "Select PSF Grid JSON", ".",
            std::vector<std::string>{"JSON", "*.json", "All Files", "*"});
    }

    if (ImGui::Button("Reload PSF Grid")) {
        sensor.loadPsfGrid();
        changed = true;
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(sensor.psfLoadStatus.c_str());

    return changed;
}
}

#ifndef NR_GPU_CODE
std::string_view Sensor::getImageSensorPath() const
{
    if (ptr())
        return DispatchCPU([](const auto* sensor) -> std::string_view {
            return sensor->imageSensorPath;
        });
    return imageSensorPath;
}

void Sensor::setImageSensorPath(std::string_view path)
{
    if (ptr()) {
        DispatchCPU([&](auto* sensor) {
            sensor->setImageSensorPath(path);
        });
    } else {
        const size_t copyCount = std::min(path.size(), sizeof(imageSensorPath) - 1);
        std::memcpy(imageSensorPath, path.data(), copyCount);
        imageSensorPath[copyCount] = '\0';
    }
}

bool Sensor::loadImageSensorDimensions()
{
    if (ptr())
        return DispatchCPU([](auto* sensor) {
            return sensor->loadImageSensorDimensions();
        });

    if (imageSensorPath[0] == '\0') {
        std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%s",
            "Sensor file path is required");
        return false;
    }

    try {
        const ross::ImageSensor loadedSensor = ross::ImageSensorReader::readFile(imageSensorPath);
        setDimensionsMm(
            loadedSensor.dimensions.width.millimeter(),
            loadedSensor.dimensions.height.millimeter());
        setResolution(loadedSensor.resolution.width, loadedSensor.resolution.height);
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

void Sensor::freeConcrete()
{
    const float previousWidth = width();
    const float previousHeight = height();
    const uint32_t previousResolutionX = resolutionX();
    const uint32_t previousResolutionY = resolutionY();
    const std::string previousImageSensorPath(getImageSensorPath());
    std::string previousImageSensorLoadStatus;
    if (ptr()) {
        previousImageSensorLoadStatus = DispatchCPU([](const auto* sensor) {
            return std::string(sensor->imageSensorLoadStatus);
        });
    } else {
        previousImageSensorLoadStatus = imageSensorLoadStatus;
    }

    if (ptr()) {
        synchronizeBeforeManagedSensorMutation("Sensor free");
        DispatchCPU([](auto* concrete) {
            using SensorType = std::remove_reference_t<decltype(*concrete)>;
            delete concrete->imageSensorDialog;
            concrete->imageSensorDialog = nullptr;
            nr::rstd::allocator<SensorType> allocator;
            allocator.destroy(concrete);
            allocator.deallocate(concrete, 1);
        });
    }

    *this = Sensor(nullptr);
    setImageSensorPath(previousImageSensorPath);
    std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%s",
        previousImageSensorLoadStatus.c_str());
    setDimensionsMm(previousWidth, previousHeight);
    setResolution(previousResolutionX, previousResolutionY);
}

void Sensor::moveConcreteFrom(Sensor& other)
{
    if (this == &other)
        return;

    freeConcrete();
    *this = other;
    other = Sensor(nullptr);
}

void Sensor::cloneConcreteFrom(const Sensor& other)
{
    if (this == &other)
        return;

    if (other.Is<ScatterPsfSensor>())
        cloneSensor<ScatterPsfSensor>(*this, other);
    else if (other.Is<GatherPsfSensor>())
        cloneSensor<GatherPsfSensor>(*this, other);
    else if (other.Is<RectangularSensor>())
        cloneSensor<RectangularSensor>(*this, other);
    else {
        freeConcrete();
        setImageSensorPath(other.getImageSensorPath());
        setDimensionsMm(other.width(), other.height());
        setResolution(other.resolutionX(), other.resolutionY());
    }
}

void Sensor::allocateRectangular()
{
    allocateSensor<RectangularSensor>(*this);
}

void Sensor::allocateScatterPsf()
{
    allocateSensor<ScatterPsfSensor>(*this);
}

void Sensor::allocateGatherPsf()
{
    allocateSensor<GatherPsfSensor>(*this);
}
#endif

bool Sensor::renderUi()
{
#ifndef NR_GPU_CODE
    if (ptr())
        return DispatchCPU([this](auto* sensor) { return sensor->renderUi(*this); });
    return renderPhysicalSensorRows(*this);
#else
    return false;
#endif
}

bool RectangularSensor::renderUi(Sensor& owner)
{
    if (!beginSensorUi("Rectangular###SensorProperties"))
        return false;

    bool changed = false;
    if (ImGui::BeginTable("SensorTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        changed |= Sensor::renderUi();
        changed |= renderSensorTypeCombo(owner, RectangularSensorTypeIndex);
        ImGui::EndTable();
    }
    ImGui::TreePop();

    return changed;
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
    return psfGrid == nullptr ? 0u : static_cast<uint32_t>(psfGrid->metadata.psfs.size());
}

bool ScatterPsfSensor::renderUi(Sensor& owner)
{
    if (!beginSensorUi("Scatter PSF###SensorProperties"))
        return false;

    bool changed = false;
    if (ImGui::BeginTable("SensorTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        changed |= Sensor::renderUi();
        changed |= renderSensorTypeCombo(owner, ScatterPsfSensorTypeIndex);
        if (owner.Is<ScatterPsfSensor>())
            changed |= renderPsfGridRows(*this);
        ImGui::EndTable();
    }
    ImGui::TreePop();

    return changed;
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
    return psfGrid == nullptr ? 0u : static_cast<uint32_t>(psfGrid->metadata.psfs.size());
}

void GatherPsfSensor::freeScratch(cudaStream_t stream) noexcept
{
    nr::rstd::deallocate_device(psfGatherBuckets, stream);
    psfGatherBuckets = nullptr;
    psfGatherBucketCapacity = 0;
}

void GatherPsfSensor::prepareFrame(const uint32_t width, const uint32_t height,
    const bool resetAccumulation, const cudaStream_t stream,
    PsfGatherBucketSample*& buckets, uint32_t& binCount)
{
    buckets = nullptr;
    binCount = psfBinCount();
    if (psfGrid == nullptr || binCount == 0)
        return;

    const size_t required = static_cast<size_t>(width) * height * binCount;
    if (required > psfGatherBucketCapacity) {
        freeScratch(stream);
        psfGatherBuckets = nr::rstd::allocate_device<PsfGatherBucketSample>(required, stream);
        psfGatherBucketCapacity = required;
    }

    buckets = psfGatherBuckets;
    if (resetAccumulation)
        NR_GPU_CHECK(cudaMemsetAsync(psfGatherBuckets, 0,
            sizeof(PsfGatherBucketSample) * psfGatherBucketCapacity, stream));
}

bool GatherPsfSensor::renderUi(Sensor& owner)
{
    if (!beginSensorUi("Gather PSF###SensorProperties"))
        return false;

    bool changed = false;
    if (ImGui::BeginTable("SensorTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        changed |= Sensor::renderUi();
        changed |= renderSensorTypeCombo(owner, GatherPsfSensorTypeIndex);
        if (owner.Is<GatherPsfSensor>())
            changed |= renderPsfGridRows(*this);
        ImGui::EndTable();
    }
    ImGui::TreePop();

    return changed;
}
