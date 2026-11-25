#include "DiffRenderer/Methods/GaussianSplatting/GaussianModel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "DiffRenderer/Methods/GaussianSplatting/NearestNeighbours.h"
#include "SplatDensification.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

constexpr int referenceIterations = 30000;
constexpr float shC0 = 0.28209479177387814f;
constexpr std::uint32_t shCount = 16;
constexpr float startOpacity = 0.1f;
constexpr float minSquaredDistance = 1e-7f;
constexpr float positionLearningRate = 1.6e-4f;
constexpr float positionFinalLearningRate = 1.6e-6f;
constexpr float featureLearningRate = 2.5e-3f;
constexpr float restLearningRateDivisor = 20.0f;
//! The paper and the first release of the reference use 0.05; the release of October 2024 changed it to 0.025.
constexpr float opacityLearningRate = 0.05f;
constexpr float scaleLearningRate = 5e-3f;
constexpr float rotationLearningRate = 1e-3f;
constexpr float gradientThreshold = 2e-4f;
constexpr float percentDense = 0.01f;
constexpr float pruneOpacity = 0.005f;
constexpr float pruneScaleFactor = 0.1f;
constexpr float resetOpacityValue = 0.01f;
constexpr int maxShDegree = 3;
constexpr double capacityGrowth = 1.1;

float getLogit(const float probability) {
    return std::log(probability / (1.0f - probability));
}

std::uint32_t grown(const std::uint64_t needed, const double factor, const std::uint64_t limit) {
    return static_cast<std::uint32_t>(std::min<std::uint64_t>(static_cast<std::uint64_t>(static_cast<double>(needed) * factor) + 1, limit));
}

}

std::uint32_t GaussianModel::getThreads(const std::uint64_t total) {
    return static_cast<std::uint32_t>(std::clamp<std::uint64_t>(total, 1, maxThreads));
}

std::expected<void, Error> GaussianModel::open(Device& target, const Scenes::Dataset& dataset, const Settings& chosen) {
    *this = GaussianModel();
    if (dataset.points.positions.empty())
        return fail(ErrorCode::InvalidArgument, "the Gaussians start from the point cloud of the dataset, and it is empty");
    device = &target;
    settings = chosen;
    const auto scale = [&](const int reference) {
        if (chosen.iterations == referenceIterations)
            return reference;
        return std::max(1, static_cast<int>(std::lround(static_cast<double>(reference) * chosen.iterations / referenceIterations)));
    };
    schedule = {chosen.iterations, scale(1000), scale(500), scale(15000), scale(100), scale(3000)};
    whiteBackground = dataset.background.x == 1.0f && dataset.background.y == 1.0f && dataset.background.z == 1.0f;
    extent = dataset.extent;
    count = static_cast<std::uint32_t>(dataset.points.positions.size());
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 6>{{{&countOutputs, &SplatDensification::countOutputs},
                                                                                              {&totalOutputs, &SplatDensification::totalOutputs},
                                                                                              {&writeSources, &SplatDensification::writeSources},
                                                                                              {&splitChildren, &SplatDensification::splitChildren},
                                                                                              {&clearFloats, &SplatDensification::clearFloats},
                                                                                              {&resetOpacities, &SplatDensification::resetOpacity}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    if (std::expected<void, Error> opened = adam.open(target); !opened)
        return opened;
    for (std::size_t group = 0; group < GroupCount; ++group)
        for (Buffer<float>* buffer : {&groups[group].parameters, &groups[group].first, &groups[group].second})
            if (std::expected<void, Error> opened = buffer->open(target, std::size_t{count} * strides[group]); !opened)
                return opened;
    if (std::expected<void, Error> opened = openWorkBuffers(count); !opened)
        return opened;
    for (std::expected<void, Error> step : {parameterSources.open(target, std::size_t{capacity} * 2), momentSources.open(target, std::size_t{capacity} * 2),
                                            children.open(target, std::size_t{capacity} * 2), newCount.open(target, 1), newCountReadback.open(target, 1)})
        if (!step)
            return step;
    return upload(dataset);
}

std::expected<void, Error> GaussianModel::openWorkBuffers(const std::uint32_t chosen) {
    capacity = chosen;
    const std::size_t size = capacity;
    for (std::expected<void, Error> step : {gradientSums.open(*device, size), gradientCounts.open(*device, size), outputs.open(*device, size),
                                            offsets.open(*device, size), scan.open(*device, size)})
        if (!step)
            return step;
    for (std::size_t group = 0; group < GroupCount; ++group)
        if (std::expected<void, Error> opened = groups[group].gradients.open(*device, size * strides[group]); !opened)
            return opened;
    return {};
}

std::expected<void, Error> GaussianModel::upload(const Scenes::Dataset& dataset) {
    const std::span<const float3> positions = dataset.points.positions;
    const std::vector<float> squaredDistances = getMeanSquaredNeighbourDistances(positions);
    std::array<std::vector<float>, GroupCount> values;
    for (std::size_t group = 0; group < GroupCount; ++group)
        values[group].assign(std::size_t{count} * strides[group], 0.0f);
    for (std::size_t index = 0; index < count; ++index) {
        const float3 position = positions[index];
        const float3 color = dataset.points.colors[index];
        const float logScale = std::log(std::sqrt(std::max(squaredDistances[index], minSquaredDistance)));
        for (std::size_t axis = 0; axis < 3; ++axis) {
            values[Means][index * 3 + axis] = (&position.x)[axis];
            values[FeaturesDc][index * 3 + axis] = ((&color.x)[axis] - 0.5f) / shC0;
            values[Scales][index * 3 + axis] = logScale;
        }
        values[Opacities][index] = getLogit(startOpacity);
        values[Rotations][index * 4] = 1.0f;
    }
    CommandList list(*device);
    for (std::size_t group = 0; group < GroupCount; ++group) {
        if (std::expected<void, Error> uploaded = groups[group].parameters.upload(values[group]); !uploaded)
            return uploaded;
        recordClear(list, groups[group].first.getPtr(), count * strides[group]);
        recordClear(list, groups[group].second.getPtr(), count * strides[group]);
    }
    recordClear(list, gradientSums.getPtr(), count);
    recordClear(list, gradientCounts.getPtr(), count);
    return submit(std::move(list));
}

std::expected<void, Error> GaussianModel::submit(CommandList&& list) const {
    std::expected<Token, Error> token = device->submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device->wait(*token);
}

std::uint32_t GaussianModel::getShDegree() const {
    return static_cast<std::uint32_t>(std::min(maxShDegree, iteration / schedule.shInterval));
}

GaussianModel::Phase GaussianModel::beginStep() {
    ++iteration;
    Phase phase;
    phase.densifying = iteration < schedule.densifyUntil;
    phase.densify = phase.densifying && iteration > schedule.densifyFrom && iteration % schedule.densifyInterval == 0;
    phase.reset = phase.densifying && (iteration % schedule.opacityResetInterval == 0 || (whiteBackground && iteration == schedule.densifyFrom));
    stepSizes = {};
    if (iteration < schedule.iterations && !phase.densify)
        setOptimizer(!phase.reset);
    return phase;
}

void GaussianModel::recordUpdate(CommandList& list) const {
    for (std::size_t group = 0; group < GroupCount; ++group) {
        const float stepSize = stepSizes[group];
        //! A step size of zero holds the group, as the opacities on a step that does not train them.
        if (stepSize == 0.0f)
            continue;
        const GroupBuffers& buffers = groups[group];
        adam.recordCorrected(list,
                             {buffers.parameters.getPtr(), buffers.gradients.getPtr(), buffers.first.getPtr(), buffers.second.getPtr(), count * strides[group]},
                             stepSize, secondCorrections[group]);
    }
}

float GaussianModel::getPositionLearningRate() const {
    const double t = std::clamp(static_cast<double>(iteration) / schedule.iterations, 0.0, 1.0);
    const double logRate = std::log(double{positionLearningRate}) * (1.0 - t) + std::log(double{positionFinalLearningRate}) * t;
    return static_cast<float>(std::exp(logRate) * extent);
}

void GaussianModel::setOptimizer(const bool opacity) {
    const std::array<float, GroupCount> learningRates{getPositionLearningRate(), featureLearningRate, featureLearningRate / restLearningRateDivisor,
                                                      opacityLearningRate,       scaleLearningRate,   rotationLearningRate};
    const Training::AdamSettings& adamSettings = adam.getSettings();
    ++adamSteps;
    if (opacity)
        ++opacityAdamSteps;
    stepSizes = {};
    secondCorrections = {};
    for (std::size_t group = 0; group < GroupCount; ++group) {
        if (group == Opacities && !opacity)
            continue;
        const std::uint32_t steps = group == Opacities ? opacityAdamSteps : adamSteps;
        const double firstCorrection = 1.0 - std::pow(double{adamSettings.beta1}, steps);
        const double secondCorrection = 1.0 - std::pow(double{adamSettings.beta2}, steps);
        stepSizes[group] = static_cast<float>(learningRates[group] * settings.learningRateScale / firstCorrection);
        secondCorrections[group] = static_cast<float>(1.0 / std::sqrt(secondCorrection));
    }
}

std::expected<void, Error> GaussianModel::finishStep(const Phase& phase) {
    if (phase.densify)
        if (std::expected<void, Error> densified = densify(); !densified)
            return densified;
    if (phase.reset)
        return resetOpacity();
    return {};
}

TileSplatting::DensifyArguments GaussianModel::getDensifyArguments() const {
    TileSplatting::DensifyArguments densify{};
    densify.parameters = {groups[Means].parameters.getPtr(),     groups[FeaturesDc].parameters.getPtr(), groups[FeaturesRest].parameters.getPtr(),
                          groups[Opacities].parameters.getPtr(), groups[Scales].parameters.getPtr(),     groups[Rotations].parameters.getPtr()};
    densify.gradientSums = gradientSums.getPtr();
    densify.gradientCounts = gradientCounts.getPtr();
    densify.outputs = outputs.getPtr();
    densify.offsets = offsets.getPtr();
    densify.newCount = newCount.getPtr();
    densify.parameterSources = parameterSources.getPtr();
    densify.momentSources = momentSources.getPtr();
    densify.children = children.getPtr();
    densify.count = count;
    densify.threads = getThreads(count);
    densify.gradientThreshold = gradientThreshold;
    densify.splitScale = percentDense * extent;
    densify.pruneOpacity = pruneOpacity;
    densify.pruneScale = iteration > schedule.opacityResetInterval ? pruneScaleFactor * extent : std::numeric_limits<float>::infinity();
    densify.resetOpacity = getLogit(resetOpacityValue);
    densify.seed = static_cast<uint>(settings.seed * 0x9e3779b9u + densifications);
    return densify;
}

void GaussianModel::recordClear(CommandList& list, const Ptr<float> destination, const std::uint32_t total) const {
    TileSplatting::DensifyArguments clear{};
    clear.destination = destination;
    clear.count = total;
    clear.threads = getThreads(total);
    list.dispatch(clearFloats, {clear.threads}, clear);
}

std::expected<void, Error> GaussianModel::densify() {
    TileSplatting::DensifyArguments densifyArguments = getDensifyArguments();
    CommandList classifying(*device);
    classifying.dispatch(countOutputs, {densifyArguments.threads}, densifyArguments);
    if (std::expected<void, Error> scanned = scan.record(classifying, outputs.getPtr(), offsets.getPtr(), count); !scanned)
        return scanned;
    classifying.dispatch(totalOutputs, {1}, densifyArguments);
    classifying.dispatch(writeSources, {densifyArguments.threads}, densifyArguments);
    classifying.download(newCount, newCountReadback);
    if (std::expected<void, Error> submitted = submit(std::move(classifying)); !submitted)
        return submitted;
    std::expected<std::span<const uint>, Error> read = newCountReadback.read();
    if (!read)
        return std::unexpected(std::move(read.error()));
    const std::uint32_t total = read->front();
    if (total == 0)
        return fail(ErrorCode::InvalidState, "the densification pruned every Gaussian");
    const std::uint32_t previousCapacity = capacity;
    capacity = total > capacity ? grown(total, capacityGrowth, maxThreads) : capacity;
    for (std::size_t group = 0; group < GroupCount; ++group)
        for (const auto& [buffer, sources] : {std::pair{&groups[group].parameters, &parameterSources}, std::pair{&groups[group].first, &momentSources},
                                              std::pair{&groups[group].second, &momentSources}})
            if (std::expected<void, Error> gathered = gather(*buffer, *sources, total, strides[group]); !gathered)
                return gathered;
    if (capacity > previousCapacity)
        if (std::expected<void, Error> opened = openWorkBuffers(capacity); !opened)
            return opened;
    CommandList gathering(*device);
    count = total;
    densifyArguments = getDensifyArguments();
    gathering.dispatch(splitChildren, {densifyArguments.threads}, densifyArguments);
    recordClear(gathering, gradientSums.getPtr(), count);
    recordClear(gathering, gradientCounts.getPtr(), count);
    if (std::expected<void, Error> submitted = submit(std::move(gathering)); !submitted)
        return submitted;
    ++densifications;
    if (parameterSources.getSize() >= std::size_t{capacity} * 2)
        return {};
    for (Buffer<uint>* buffer : {&parameterSources, &momentSources, &children})
        if (std::expected<void, Error> opened = buffer->open(*device, std::size_t{capacity} * 2); !opened)
            return opened;
    return {};
}

std::expected<void, Error> GaussianModel::gather(Buffer<float>& buffer, const Buffer<uint>& sources, const std::uint32_t total, const std::uint32_t stride) {
    Buffer<float> gathered;
    if (std::expected<void, Error> opened = gathered.open(*device, std::size_t{capacity} * stride); !opened)
        return opened;
    CommandList list(*device);
    adam.recordGather(list, buffer.getPtr(), gathered.getPtr(), sources.getPtr(), total, stride);
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return submitted;
    buffer = std::move(gathered);
    return {};
}

std::expected<void, Error> GaussianModel::resetOpacity() const {
    TileSplatting::DensifyArguments reset = getDensifyArguments();
    reset.firstMoments = groups[Opacities].first.getPtr();
    reset.secondMoments = groups[Opacities].second.getPtr();
    CommandList list(*device);
    list.dispatch(resetOpacities, {reset.threads}, reset);
    return submit(std::move(list));
}

std::expected<TrainedGaussians, Error> GaussianModel::readGaussians() const {
    std::array<Readback<float>, GroupCount> readbacks;
    CommandList list(*device);
    for (std::size_t group = 0; group < GroupCount; ++group) {
        if (std::expected<void, Error> opened = readbacks[group].open(*device, std::size_t{count} * strides[group]); !opened)
            return std::unexpected(std::move(opened.error()));
        list.download(groups[group].parameters, readbacks[group]);
    }
    if (std::expected<void, Error> submitted = submit(std::move(list)); !submitted)
        return std::unexpected(std::move(submitted.error()));
    std::array<std::span<const float>, GroupCount> values;
    for (std::size_t group = 0; group < GroupCount; ++group) {
        std::expected<std::span<const float>, Error> read = readbacks[group].read();
        if (!read)
            return std::unexpected(std::move(read.error()));
        values[group] = *read;
    }
    TrainedGaussians gaussians;
    gaussians.shDegree = getShDegree();
    gaussians.means.assign(values[Means].begin(), values[Means].end());
    gaussians.scales.resize(values[Scales].size());
    std::ranges::transform(values[Scales], gaussians.scales.begin(), [](const float logScale) { return std::exp(logScale); });
    gaussians.opacities.resize(count);
    std::ranges::transform(values[Opacities], gaussians.opacities.begin(), [](const float logit) { return 1.0f / (1.0f + std::exp(-logit)); });
    gaussians.quaternions.resize(std::size_t{count} * 4);
    gaussians.sh.resize(std::size_t{count} * shCount * 3);
    for (std::size_t index = 0; index < count; ++index) {
        const std::span<const float> wxyz = values[Rotations].subspan(index * 4, 4);
        const float norm = std::sqrt(wxyz[0] * wxyz[0] + wxyz[1] * wxyz[1] + wxyz[2] * wxyz[2] + wxyz[3] * wxyz[3]);
        for (std::size_t component = 0; component < 4; ++component)
            gaussians.quaternions[index * 4 + component] = wxyz[(component + 1) % 4] / norm;
        const std::span<float> sh = std::span(gaussians.sh).subspan(index * shCount * 3, shCount * 3);
        std::ranges::copy(values[FeaturesDc].subspan(index * 3, 3), sh.begin());
        std::ranges::copy(values[FeaturesRest].subspan(index * (shCount - 1) * 3, (shCount - 1) * 3), sh.begin() + 3);
    }
    return gaussians;
}

Statistics GaussianModel::getStatistics() const {
    std::size_t floats = 0;
    for (const std::uint32_t stride : strides)
        floats += stride;
    return {count, std::size_t{count} * floats * sizeof(float)};
}

}
