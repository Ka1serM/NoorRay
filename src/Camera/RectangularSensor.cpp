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
#include "CUDA/ManagedMemory.h"
#include "Log.h"
#include "UI/ImGuiManager.h"
#include "libross/foundation/gpu/types/Allocator.h"
#include "libross/imaging/imagesensor/ImageSensorReader.h"
#include "portable-file-dialogs.h"

namespace {
constexpr int RectangularSensorTypeIndex = 0;
constexpr int ScatterPsfSensorTypeIndex = 1;
constexpr int GatherPsfSensorTypeIndex = 2;

constexpr std::array<const char*, 3> SensorTypeNames{
    "Rectangular", "Scatter PSF", "Gather PSF"};

template <typename SensorType>
void allocateSensor(Sensor& sensor)
{
    nr::synchronizeBeforeManagedMutation("Sensor allocation");

    std::string previousPsfGridPath;
    if (const auto* scatter = sensor.CastOrNullptr<ScatterPsfSensor>())
        previousPsfGridPath = scatter->psfGridPath;
    if (const auto* gather = sensor.CastOrNullptr<GatherPsfSensor>())
        previousPsfGridPath = gather->psfGridPath;

    sensor.freeConcrete();

    nr::rstd::allocator<SensorType> allocator;
    SensorType* concrete = allocator.allocate(1);
    allocator.construct(concrete);
    if constexpr (requires { concrete->psfGrid; }) {
        concrete->psfGridPath = previousPsfGridPath;
        if (!concrete->psfGridPath.empty())
            concrete->loadPsfGrid();
    }
    static_cast<TaggedSensor&>(sensor) = TaggedSensor(concrete);
}

template <typename SensorType>
void cloneSensor(Sensor& destination, const Sensor& source)
{
    destination.freeConcrete();

    nr::rstd::allocator<SensorType> allocator;
    SensorType* concrete = allocator.allocate(1);
    allocator.construct(concrete);

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

    static_cast<TaggedSensor&>(destination) = TaggedSensor(concrete);
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
    const float reloadButtonWidth = ImGui::CalcTextSize("Reload").x + ImGui::GetStyle().FramePadding.x * 2.0f;

    ImGuiManager::tableRowLabel("Sensor File");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - selectButtonWidth - reloadButtonWidth
        - ImGui::GetStyle().ItemSpacing.x * 2.0f);
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
    ImGui::SameLine();
    if (ImGui::Button("Reload##ImageSensor", ImVec2(reloadButtonWidth, 0))) {
        changed |= sensor.loadImageSensorDimensions();
    }

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

    int resolutionX = static_cast<int>(sensor.resolutionX());
    int resolutionY = static_cast<int>(sensor.resolutionY());
    ImGuiManager::tableRowLabel("Resolution X");
    if (ImGui::InputInt("##ResolutionX", &resolutionX, 0, 0, ImGuiInputTextFlags_CharsDecimal)
        && resolutionX > 0) {
        sensor.setResolution(static_cast<uint32_t>(resolutionX), sensor.resolutionY());
        changed = true;
    }
    ImGuiManager::tableRowLabel("Resolution Y");
    if (ImGui::InputInt("##ResolutionY", &resolutionY, 0, 0, ImGuiInputTextFlags_CharsDecimal)
        && resolutionY > 0) {
        sensor.setResolution(sensor.resolutionX(), static_cast<uint32_t>(resolutionY));
        changed = true;
    }

    return changed;
}

template <typename PsfSensor>
void freePsfGrid(PsfSensor& sensor)
{
    if (sensor.psfGrid == nullptr)
        return;

    nr::synchronizeBeforeManagedMutation("PSF grid free");

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

void Sensor::freeConcrete()
{
    if (!ptr())
        return;

    nr::synchronizeBeforeManagedMutation("Sensor free");
    DispatchCPU([](auto* concrete) {
        using SensorType = std::remove_reference_t<decltype(*concrete)>;
        nr::rstd::allocator<SensorType> allocator;
        allocator.destroy(concrete);
        allocator.deallocate(concrete, 1);
    });

    static_cast<TaggedSensor&>(*this) = TaggedSensor(nullptr);
}

void Sensor::moveConcreteFrom(Sensor& other)
{
    if (this == &other)
        return;

    freeConcrete();
    *this = other;
    static_cast<TaggedSensor&>(other) = TaggedSensor(nullptr);
}

void Sensor::cloneConcreteFrom(const Sensor& other)
{
    if (this == &other)
        return;

    copyPhysicalFrom(other);
    setImageSensorPath(other.getImageSensorPath());
    std::snprintf(imageSensorLoadStatus, sizeof(imageSensorLoadStatus), "%s",
        other.imageSensorLoadStatus);

    if (!other) {
        allocateRectangular();
    } else if (other.Is<ScatterPsfSensor>())
        cloneSensor<ScatterPsfSensor>(*this, other);
    else if (other.Is<GatherPsfSensor>())
        cloneSensor<GatherPsfSensor>(*this, other);
    else if (other.Is<RectangularSensor>())
        cloneSensor<RectangularSensor>(*this, other);
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
#endif

bool Sensor::renderUi()
{
#ifndef NR_GPU_CODE
    if (auto* rectangular = CastOrNullptr<RectangularSensor>())
        return rectangular->renderUi(*this);
    if (auto* scatter = CastOrNullptr<ScatterPsfSensor>())
        return scatter->renderUi(*this);
    if (auto* gather = CastOrNullptr<GatherPsfSensor>())
        return gather->renderUi(*this);
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
        changed |= renderSensorTypeCombo(owner, RectangularSensorTypeIndex);
        changed |= renderPhysicalSensorRows(owner);
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
        changed |= renderSensorTypeCombo(owner, ScatterPsfSensorTypeIndex);
        changed |= renderPhysicalSensorRows(owner);
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
    if (psfGrid == nullptr || binCount == 0)
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

bool GatherPsfSensor::renderUi(Sensor& owner)
{
    if (!beginSensorUi("Gather PSF###SensorProperties"))
        return false;

    bool changed = false;
    if (ImGui::BeginTable("SensorTable", 2, ImGuiTableFlags_SizingStretchProp)) {
        changed |= renderSensorTypeCombo(owner, GatherPsfSensorTypeIndex);
        changed |= renderPhysicalSensorRows(owner);
        if (owner.Is<GatherPsfSensor>())
            changed |= renderPsfGridRows(*this);
        ImGui::EndTable();
    }
    ImGui::TreePop();

    return changed;
}
