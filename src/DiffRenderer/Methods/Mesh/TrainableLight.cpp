#include "DiffRenderer/Methods/Mesh/TrainableLight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

#include "EnvironmentLighting.slang.h"

namespace DiffRenderer::Methods {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t levelCount = EnvironmentLight::levelCount;
constexpr float lobeFraction = 0.99f;
constexpr std::size_t cutoffSamples = 100'000;
//! nvdiffrec multiplies the gradient of the light by 64 before its Adam step.
constexpr float gradientScale = 64.0f;
constexpr float whiteWeight = 0.005f;

float getLevelRoughness(const std::uint32_t level) {
    if (level + 1 == levelCount)
        return 1.0f;
    return EnvironmentLight::minimumRoughness
           + (EnvironmentLight::maximumRoughness - EnvironmentLight::minimumRoughness) * static_cast<float>(level) / static_cast<float>(levelCount - 2);
}

//! The cosine where the cumulative GGX lobe over a uniform angle grid reaches 99%, as __ndfBounds of nvdiffrec.
float getLobeCutoff(const float roughness) {
    const double alphaSquared = std::pow(static_cast<double>(roughness), 4.0);
    std::vector<double> cumulative(cutoffSamples);
    std::vector<double> cosines(cutoffSamples);
    double sum = 0.0;
    for (std::size_t index = 0; index < cutoffSamples; ++index) {
        const double cosine = std::cos(std::numbers::pi / 2.0 * static_cast<double>(index) / static_cast<double>(cutoffSamples - 1));
        const double d = (cosine * alphaSquared - cosine) * cosine + 1.0;
        sum += alphaSquared / (d * d * std::numbers::pi);
        cumulative[index] = sum;
        cosines[index] = cosine;
    }
    const auto found = std::ranges::find_if(cumulative, [&](const double value) { return value >= sum * lobeFraction; });
    return static_cast<float>(cosines[static_cast<std::size_t>(found - cumulative.begin())]);
}

std::expected<void, Error> submit(Device& device, CommandList&& list) {
    std::expected<Token, Error> token = device.submit(std::move(list));
    if (!token)
        return std::unexpected(std::move(token.error()));
    return device.wait(*token);
}

}

std::uint32_t TrainableLight::getTexelCount(const std::uint32_t level) const {
    const std::uint32_t size = arguments.light.baseSize >> level;
    return 6 * size * size;
}

std::uint32_t TrainableLight::getTotalTexelCount() const {
    std::uint32_t count = 0;
    for (std::uint32_t level = 0; level < levelCount; ++level)
        count += getTexelCount(level);
    return count;
}

std::size_t TrainableLight::getByteCount() const {
    return std::size_t{getTexelCount(0)} * 3 * sizeof(float);
}

std::expected<void, Error> TrainableLight::open(Device& target, const std::uint32_t baseSize) {
    device = &target;
    for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 9>{{{&downsample, &EnvironmentLighting::downsample},
                                                                                              {&filterSpecular, &EnvironmentLighting::filterSpecular},
                                                                                              {&filterDiffuse, &EnvironmentLighting::filterDiffuse},
                                                                                              {&clearGradients, &EnvironmentLighting::clearGradients},
                                                                                              {&filterSpecularBackward, &EnvironmentLighting::filterSpecularBackward},
                                                                                              {&filterDiffuseBackward, &EnvironmentLighting::filterDiffuseBackward},
                                                                                              {&upsampleGradient, &EnvironmentLighting::upsampleGradient},
                                                                                              {&finishBaseGradient, &EnvironmentLighting::finishBaseGradient},
                                                                                              {&clampBase, &EnvironmentLighting::clampBase}}})
        if (std::expected<void, Error> opened = pipeline->open(target, *shader); !opened)
            return opened;
    arguments = {};
    arguments.light.baseSize = baseSize;
    const std::size_t total = getTotalTexelCount();
    const std::size_t last = getTexelCount(levelCount - 1);
    const std::size_t base = std::size_t{getTexelCount(0)} * 3;
    for (std::expected<void, Error> step : {levels.open(target, total * 3), specular.open(target, total), diffuse.open(target, last * 3),
                                            specularGradient.open(target, total * 3), diffuseGradient.open(target, last * 3),
                                            levelGradient.open(target, total * 3), first.open(target, base), second.open(target, base)})
        if (!step)
            return step;
    std::array<float, levelCount> cutoffs{};
    for (std::uint32_t level = 0; level < levelCount; ++level)
        cutoffs[level] = getLobeCutoff(getLevelRoughness(level));
    EnvironmentLight::Light& light = arguments.light;
    light.levels = levels.getPtr();
    light.specular = specular.getPtr();
    light.diffuse = diffuse.getPtr();
    light.specularGradient = specularGradient.getPtr();
    light.diffuseGradient = diffuseGradient.getPtr();
    light.levelGradient = levelGradient.getPtr();
    light.cutoffLow = {cutoffs[0], cutoffs[1], cutoffs[2], cutoffs[3]};
    light.cutoffHigh = {cutoffs[4], cutoffs[5]};
    arguments.gradientScale = gradientScale;
    arguments.whiteWeight = whiteWeight;
    return reset(0.5f);
}

void TrainableLight::close() {
    *this = TrainableLight();
}

std::expected<void, Error> TrainableLight::reset(const float value) {
    const std::vector<float> values(levels.getSize(), value);
    if (std::expected<void, Error> uploaded = levels.upload(values); !uploaded)
        return uploaded;
    return restartMoments();
}

std::expected<void, Error> TrainableLight::restartMoments() {
    const std::vector<float> zeros(first.getSize(), 0.0f);
    step = 0;
    for (std::expected<void, Error> uploaded : {first.upload(zeros), second.upload(zeros)})
        if (!uploaded)
            return uploaded;
    return {};
}

void TrainableLight::dispatchLevels(CommandList& list, const ComputePipeline& pipeline, const std::uint32_t firstLevel, const std::uint32_t lastLevel,
                                    const bool down) const {
    EnvironmentLight::Arguments levelArguments = arguments;
    for (std::uint32_t index = 0; index <= lastLevel - firstLevel; ++index) {
        const std::uint32_t level = down ? lastLevel - index : firstLevel + index;
        levelArguments.light.level = level;
        list.dispatch(pipeline, {getTexelCount(level)}, levelArguments);
    }
}

void TrainableLight::recordBuild(CommandList& list) const {
    dispatchLevels(list, downsample, 1, levelCount - 1, false);
    dispatchLevels(list, filterSpecular, 0, levelCount - 1, false);
    list.dispatch(filterDiffuse, {getTexelCount(levelCount - 1)}, arguments);
}

void TrainableLight::recordClearGradients(CommandList& list) const {
    list.dispatch(clearGradients, {getTotalTexelCount() * 3}, arguments);
}

void TrainableLight::recordBackward(CommandList& list) const {
    dispatchLevels(list, filterSpecularBackward, 0, levelCount - 1, false);
    list.dispatch(filterDiffuseBackward, {getTexelCount(levelCount - 1)}, arguments);
    dispatchLevels(list, upsampleGradient, 0, levelCount - 2, true);
    list.dispatch(finishBaseGradient, {getTexelCount(0)}, arguments);
}

void TrainableLight::recordStep(CommandList& list, const Training::GpuAdam& adam, const float learningRate) {
    const std::uint32_t count = getTexelCount(0) * 3;
    adam.record(list, {levels.getPtr(), levelGradient.getPtr(), first.getPtr(), second.getPtr(), count, learningRate}, ++step);
    list.dispatch(clampBase, {count}, arguments);
}

std::expected<void, Error> TrainableLight::downloadBase(std::vector<float>& values) const {
    values.resize(std::size_t{getTexelCount(0)} * 3);
    std::vector<float> all(levels.getSize());
    if (std::expected<void, Error> downloaded = levels.download(all); !downloaded)
        return downloaded;
    std::copy_n(all.begin(), values.size(), values.begin());
    return {};
}

}
