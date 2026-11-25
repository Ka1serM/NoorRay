#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <print>
#include <random>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Core/GaussianProxies.h"
#include "DiffRenderer/Methods/GaussianTracing/Shaders/GaussianRays.h"
#include "GaussianRays.slang.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t size = 64;
constexpr std::uint32_t count = 24;
constexpr std::uint32_t gradientStride = 59;
constexpr int trials = 8;
constexpr float epsilon = 5e-4f;
constexpr double tolerance = 0.005;
constexpr float jumpThreshold = 5e-5f;

struct Group {
    std::string_view name;
    std::uint32_t stride;
    float center;
    float spread;
};

constexpr std::array<Group, 6> groups{{{"means", 3, 0.0f, 0.5f},
                                       {"featuresDc", 3, 0.4f, 0.8f},
                                       {"featuresRest", 45, 0.0f, 0.1f},
                                       {"opacities", 1, 0.0f, 1.0f},
                                       {"scales", 3, -1.2f, 0.4f},
                                       {"rotations", 4, 0.3f, 1.0f}}};

class RayKernel {
public:
    Result open(Device& target) {
        device = &target;
        for (auto [pipeline, shader] : std::array<std::pair<ComputePipeline*, const Shader*>, 4>{{{&clear, &GaussianRays::clearGradients},
                                                                                                  {&forward, &GaussianRays::traceForward},
                                                                                                  {&backward, &GaussianRays::traceBackward},
                                                                                                  {&gather, &GaussianRays::gatherGradients}}})
            if (Result opened = check(pipeline->open(target, *shader)); !opened)
                return opened;
        for (std::size_t group = 0; group < groups.size(); ++group)
            for (std::expected<void, Error> step : {parameters[group].open(target, std::size_t{count} * groups[group].stride),
                                                    gradients[group].open(target, std::size_t{count} * groups[group].stride),
                                                    readbacks[group].open(target, std::size_t{count} * groups[group].stride)})
                if (!step)
                    return check(step);
        const std::size_t values = std::size_t{size} * size * 3;
        for (std::expected<void, Error> step : {image.open(target, values), radiance.open(target, values), imageGradient.open(target, values),
                                                imageReadback.open(target, values), rayGradients.open(target, std::size_t{count} * gradientStride),
                                                sums.open(target, count), counts.open(target, count), proxies.open(target, count)})
            if (!step)
                return check(step);
        return {};
    }

    Result setWeights(const std::span<const float> weights) { return check(imageGradient.upload(weights)); }

    std::expected<std::vector<float>, std::string> render(const std::array<std::vector<float>, 6>& values) {
        for (std::size_t group = 0; group < groups.size(); ++group)
            if (Result uploaded = check(parameters[group].upload(std::span<const float>(values[group]))); !uploaded)
                return std::unexpected(uploaded.error());
        Instances::Arguments instances{};
        instances.means = parameters[0].getPtr();
        instances.quaternions = parameters[5].getPtr();
        instances.logScales = parameters[4].getPtr();
        instances.opacityLogits = parameters[3].getPtr();
        instances.count = count;
        CommandList writing(*device);
        proxies.recordInstances(writing, instances);
        if (Result submitted = submit(*device, std::move(writing)); !submitted)
            return std::unexpected(submitted.error());
        if (Result built = check(proxies.rebuild(count)); !built)
            return std::unexpected(built.error());
        CommandList list(*device);
        list.dispatch(forward, {size, size}, getArguments());
        list.download(image, imageReadback);
        if (Result submitted = submit(*device, std::move(list)); !submitted)
            return std::unexpected(submitted.error());
        const std::expected<std::span<const float>, Error> pixels = imageReadback.read();
        if (!pixels)
            return std::unexpected(pixels.error().message);
        return std::vector<float>(pixels->begin(), pixels->end());
    }

    std::expected<std::array<std::vector<float>, 6>, std::string> differentiate() {
        CommandList list(*device);
        list.dispatch(clear, {count * gradientStride}, getArguments(count * gradientStride));
        list.dispatch(backward, {size, size}, getArguments());
        list.dispatch(gather, {count}, getArguments(count));
        for (std::size_t group = 0; group < groups.size(); ++group)
            list.download(gradients[group], readbacks[group]);
        if (Result submitted = submit(*device, std::move(list)); !submitted)
            return std::unexpected(submitted.error());
        std::array<std::vector<float>, 6> result;
        for (std::size_t group = 0; group < groups.size(); ++group) {
            const std::expected<std::span<const float>, Error> values = readbacks[group].read();
            if (!values)
                return std::unexpected(values.error().message);
            result[group].assign(values->begin(), values->end());
        }
        return result;
    }

private:
    static GaussianRays::Parameters getPointers(const std::array<Buffer<float>, 6>& buffers) {
        return {buffers[0].getPtr(), buffers[1].getPtr(), buffers[2].getPtr(), buffers[3].getPtr(), buffers[4].getPtr(), buffers[5].getPtr()};
    }

    GaussianRays::Arguments getArguments(const std::uint32_t threads = 1) const {
        GaussianRays::Arguments arguments{};
        arguments.view = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 3}}, {64.0f, 64.0f, 32.0f, 32.0f}, {0, 0, -3, 0}, {0.1f, 0.2f, 0.3f, 0}, size, size, 0, size};
        arguments.parameters = getPointers(parameters);
        arguments.gradients = getPointers(gradients);
        arguments.rayGradients = rayGradients.getPtr();
        arguments.tlas = proxies.getTlas();
        arguments.image = image.getPtr();
        arguments.radiance = radiance.getPtr();
        arguments.imageGradient = imageGradient.getPtr();
        arguments.gradientSums = sums.getPtr();
        arguments.gradientCounts = counts.getPtr();
        arguments.count = count;
        arguments.threads = threads;
        arguments.shDegree = 3;
        return arguments;
    }

    Device* device = nullptr;
    ComputePipeline clear;
    ComputePipeline forward;
    ComputePipeline backward;
    ComputePipeline gather;
    Core::GaussianProxies proxies;
    std::array<Buffer<float>, 6> parameters;
    std::array<Buffer<float>, 6> gradients;
    std::array<Readback<float>, 6> readbacks;
    Buffer<float> image;
    Buffer<float> radiance;
    Buffer<float> imageGradient;
    Buffer<float> sums;
    Buffer<float> counts;
    Buffer<Atomic<float>> rayGradients;
    Readback<float> imageReadback;
};

//! A jump shows as a second difference far above the curvature of a smooth pixel; a masked channel masks its whole pixel.
std::size_t maskJumps(std::vector<float>& weights, const std::span<const float> high, const std::span<const float> middle, const std::span<const float> low) {
    std::size_t masked = 0;
    for (std::size_t pixel = 0; pixel < weights.size(); pixel += 3) {
        bool jumps = false;
        for (std::size_t index = pixel; index < pixel + 3; ++index)
            jumps |= std::abs(high[index] - 2.0f * middle[index] + low[index]) > jumpThreshold;
        if (!jumps)
            continue;
        std::fill_n(weights.begin() + static_cast<std::ptrdiff_t>(pixel), 3, 0.0f);
        ++masked;
    }
    return masked;
}

double sumDifference(const std::span<const float> high, const std::span<const float> low, const std::span<const float> weights) {
    double sum = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index)
        sum += (static_cast<double>(high[index]) - static_cast<double>(low[index])) * static_cast<double>(weights[index]);
    return sum;
}

}

//! The analytic derivative along random directions of each parameter group against central differences, without the pixels that jump at an order swap or the
//! cutoff.
Result testGaussianMethod(Device& device) {
    RayKernel kernel;
    if (Result opened = kernel.open(device); !opened)
        return opened;
    std::mt19937 random(7);
    std::normal_distribution<float> normal;
    std::array<std::vector<float>, 6> values;
    for (std::size_t group = 0; group < groups.size(); ++group) {
        values[group].resize(std::size_t{count} * groups[group].stride);
        for (float& value : values[group])
            value = groups[group].center + groups[group].spread * normal(random);
    }
    std::vector<float> weights(std::size_t{size} * size * 3);
    for (std::size_t index = 0; index < weights.size(); ++index)
        weights[index] = 1.0f + 0.5f * std::sin(static_cast<float>(index) * 0.013f);
    const auto middleImage = kernel.render(values);
    if (!middleImage)
        return std::unexpected(middleImage.error());
    std::string failures;
    for (std::size_t group = 0; group < groups.size(); ++group) {
        double error = 0.0;
        double scale = 0.0;
        std::size_t masked = 0;
        for (int trial = 0; trial < trials; ++trial) {
            std::array<std::vector<float>, 6> high = values;
            std::array<std::vector<float>, 6> low = values;
            std::vector<float> direction(values[group].size());
            for (std::size_t index = 0; index < direction.size(); ++index) {
                direction[index] = normal(random);
                high[group][index] += epsilon * direction[index];
                low[group][index] -= epsilon * direction[index];
            }
            const auto highImage = kernel.render(high);
            if (!highImage)
                return std::unexpected(highImage.error());
            const auto lowImage = kernel.render(low);
            if (!lowImage)
                return std::unexpected(lowImage.error());
            std::vector<float> smoothWeights = weights;
            masked += maskJumps(smoothWeights, *highImage, *middleImage, *lowImage);
            if (Result uploaded = kernel.setWeights(smoothWeights); !uploaded)
                return uploaded;
            if (const auto rendered = kernel.render(values); !rendered)
                return std::unexpected(rendered.error());
            const auto gradient = kernel.differentiate();
            if (!gradient)
                return std::unexpected(gradient.error());
            double analytic = 0.0;
            for (std::size_t index = 0; index < direction.size(); ++index)
                analytic += static_cast<double>((*gradient)[group][index]) * static_cast<double>(direction[index]);
            const double numeric = sumDifference(*highImage, *lowImage, smoothWeights) / (2.0 * static_cast<double>(epsilon));
            error += (analytic - numeric) * (analytic - numeric);
            scale += numeric * numeric;
        }
        const double relative = std::sqrt(error / scale);
        const double share = static_cast<double>(masked) / (static_cast<double>(trials) * size * size);
        std::println("{:<14} relative error {:.4f}, {:.2f} % of the pixels jump", groups[group].name, relative, share * 100.0);
        if (!(relative < tolerance))
            failures += std::format("{} {:.4f} ", groups[group].name, relative);
    }
    if (!failures.empty())
        return std::unexpected("the gradient differs from finite differences: " + failures);
    return {};
}

}
