#include <cmath>
#include <format>
#include <print>
#include <random>
#include <vector>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Training/GpuAdam.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t count = 100003;
constexpr int steps = 5;
constexpr float learningRate = 0.01f;

std::expected<std::vector<float>, std::string> download(Device& device, const Buffer<float>& buffer) {
    Readback<float> readback;
    if (Result opened = check(readback.open(device, buffer.getSize())); !opened)
        return std::unexpected(opened.error());
    CommandList list(device);
    list.download(buffer, readback);
    if (Result submitted = submit(device, std::move(list)); !submitted)
        return std::unexpected(submitted.error());
    std::expected<std::span<const float>, Error> values = readback.read();
    if (!values)
        return std::unexpected(values.error().message);
    return std::vector<float>(values->begin(), values->end());
}

}

Result testAdam(Device& device) {
    Training::GpuAdam adam;
    std::array<Buffer<float>, 4> buffers;
    if (Result opened = check(adam.open(device)); !opened)
        return opened;
    for (Buffer<float>& buffer : buffers)
        if (Result opened = check(buffer.open(device, count)); !opened)
            return opened;
    std::mt19937 random(11);
    std::normal_distribution<float> normal;
    std::vector<float> parameters(count);
    for (float& parameter : parameters)
        parameter = normal(random);
    const std::vector<float> zeros(count, 0.0f);
    for (std::expected<void, Error> step : {buffers[0].upload(parameters), buffers[2].upload(zeros), buffers[3].upload(zeros)})
        if (Result result = check(step); !result)
            return result;
    std::vector<double> expected(parameters.begin(), parameters.end());
    std::vector<double> first(count);
    std::vector<double> second(count);
    const Training::AdamSettings& settings = adam.getSettings();
    std::vector<float> gradients(count);
    for (int step = 1; step <= steps; ++step) {
        for (float& gradient : gradients)
            gradient = normal(random) * (random() % 7 == 0 ? 0.0f : 1e-3f);
        CommandList list(device);
        list.upload(buffers[1], std::span<const float>(gradients));
        adam.record(list, {buffers[0].getPtr(), buffers[1].getPtr(), buffers[2].getPtr(), buffers[3].getPtr(), count, learningRate},
                    static_cast<std::uint32_t>(step));
        if (Result submitted = submit(device, std::move(list)); !submitted)
            return submitted;
        const double firstCorrection = 1.0 - std::pow(double{settings.beta1}, step);
        const double secondCorrection = 1.0 - std::pow(double{settings.beta2}, step);
        for (std::size_t index = 0; index < count; ++index) {
            first[index] = settings.beta1 * first[index] + (1.0 - settings.beta1) * gradients[index];
            second[index] = settings.beta2 * second[index] + (1.0 - settings.beta2) * double{gradients[index]} * gradients[index];
            expected[index] -= learningRate / firstCorrection * first[index] / (std::sqrt(second[index] / secondCorrection) + settings.epsilon);
        }
    }
    std::expected<std::vector<float>, std::string> result = download(device, buffers[0]);
    if (!result)
        return std::unexpected(result.error());
    double error = 0.0;
    for (std::size_t index = 0; index < count; ++index)
        error = std::max(error, std::abs((*result)[index] - expected[index]));
    std::println("adam: {} parameters, {} steps, max abs error against a double CPU reference {:.3e}", count, steps, error);
    if (error > 1e-5)
        return std::unexpected("adam differs from the reference");

    constexpr std::uint32_t stride = 3;
    constexpr std::uint32_t gathered = 50000;
    std::vector<uint> indices(gathered);
    for (std::uint32_t index = 0; index < gathered; ++index)
        indices[index] = index % 10 == 9 ? Training::GpuAdam::noSource : static_cast<uint>(random() % (count / stride));
    Buffer<uint> indexBuffer;
    Buffer<float> destination;
    for (std::expected<void, Error> step : {indexBuffer.open(device, gathered), destination.open(device, gathered * stride), indexBuffer.upload(indices)})
        if (Result opened = check(step); !opened)
            return opened;
    CommandList list(device);
    adam.recordGather(list, buffers[0].getPtr(), destination.getPtr(), indexBuffer.getPtr(), gathered, stride);
    if (Result submitted = submit(device, std::move(list)); !submitted)
        return submitted;
    std::expected<std::vector<float>, std::string> copied = download(device, destination);
    if (!copied)
        return std::unexpected(copied.error());
    for (std::uint32_t entry = 0; entry < gathered; ++entry)
        for (std::uint32_t component = 0; component < stride; ++component) {
            const float wanted = indices[entry] == Training::GpuAdam::noSource ? 0.0f : (*result)[indices[entry] * stride + component];
            if ((*copied)[entry * stride + component] != wanted)
                return std::unexpected(std::format("the gather differs at entry {}", entry));
        }
    std::println("gather: {} entries of {} floats with duplicates and new zero entries: correct", gathered, stride);
    return {};
}

}
