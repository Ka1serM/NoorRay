#include <algorithm>
#include <cmath>
#include <format>
#include <print>
#include <random>
#include <vector>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Core/Image.h"
#include "DiffRenderer/Core/Metrics.h"
#include "DiffRenderer/Training/Loss.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t width = 32;
constexpr std::uint32_t height = 24;
constexpr std::size_t valueCount = std::size_t{width} * height * 3;

struct LossBench {
    Device* device = nullptr;
    Training::Loss loss;
    Buffer<float> image;
    Buffer<uint> target;
    Buffer<float> gradient;
    Readback<float> gradientReadback;

    std::expected<void, Error> open(Device& opened, const Training::LossKind kind) {
        device = &opened;
        for (std::expected<void, Error> step : {loss.open(opened, width, height, kind), image.open(opened, valueCount), target.open(opened, valueCount / 3),
                                                gradient.open(opened, valueCount), gradientReadback.open(opened, valueCount)})
            if (!step)
                return step;
        return {};
    }

    std::expected<double, std::string> evaluate(const std::vector<float>& values, std::vector<float>* gradients) {
        CommandList list(*device);
        list.upload(image, std::span<const float>(values));
        loss.record(list, image.getPtr(), target.getPtr(), gradient.getPtr(), width, height);
        loss.recordReadback(list);
        if (gradients)
            list.download(gradient, gradientReadback);
        if (Result submitted = submit(*device, std::move(list)); !submitted)
            return std::unexpected(submitted.error());
        std::expected<double, Error> value = loss.read();
        if (!value)
            return std::unexpected(value.error().message);
        if (gradients) {
            std::expected<std::span<const float>, Error> read = gradientReadback.read();
            if (!read)
                return std::unexpected(read.error().message);
            gradients->assign(read->begin(), read->end());
        }
        return *value;
    }
};

Core::Image toImage(const std::vector<float>& values) {
    Core::Image image(width, height);
    std::ranges::copy(values, image.getPixels().begin());
    return image;
}

double getMeanAbsoluteError(const std::vector<float>& values, const Core::Image& reference) {
    double sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index)
        sum += std::abs(double{values[index]} - reference.getPixels()[index]);
    return sum / static_cast<double>(values.size());
}

double getCpuLoss(const std::vector<float>& values, const Core::Image& reference, const Training::LossKind kind) {
    double l1 = 0.0;
    double squares = 0.0;
    const std::span<const float> targets = reference.getPixels();
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double difference = double{values[index]} - targets[index];
        l1 += std::abs(difference);
        squares += difference * difference;
    }
    const double count = static_cast<double>(values.size());
    if (kind == Training::LossKind::L2)
        return squares / count;
    return 0.8 * l1 / count + 0.2 * (1.0 - Core::getSsim(toImage(values), reference));
}

Result testKind(Device& device, const Training::LossKind kind, const std::string_view name) {
    std::mt19937 random(7);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> values(valueCount);
    std::vector<uint> packed(valueCount / 3);
    Core::Image reference(width, height);
    for (std::size_t pixel = 0; pixel < packed.size(); ++pixel) {
        packed[pixel] = 0xff000000u;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const auto byte = static_cast<uint>(random() % 256);
            packed[pixel] |= byte << (8 * channel);
            reference.getPixels()[pixel * 3 + channel] = static_cast<float>(byte) / 255.0f;
            values[pixel * 3 + channel] = std::clamp(reference.getPixels()[pixel * 3 + channel] + 0.3f * (uniform(random) - 0.5f), 0.0f, 1.0f);
        }
    }
    LossBench bench;
    if (Result opened = check(bench.open(device, kind)); !opened)
        return opened;
    if (Result uploaded = check(bench.target.upload(packed)); !uploaded)
        return uploaded;
    std::vector<float> gradients;
    const std::expected<double, std::string> value = bench.evaluate(values, &gradients);
    if (!value)
        return std::unexpected(value.error());
    const double cpuValue = getCpuLoss(values, reference, kind);
    std::println("{}: GPU loss {:.8f}, CPU loss {:.8f}, difference {:.2e}", name, *value, cpuValue, std::abs(*value - cpuValue));
    if (kind == Training::LossKind::L1Ssim) {
        const double ssim = Core::getSsim(toImage(values), reference);
        const double gpuSsim = 1.0 - (*value - 0.8 * getMeanAbsoluteError(values, reference)) / 0.2;
        std::println("{}: CPU SSIM {:.8f}, GPU SSIM {:.8f}, difference {:.2e}", name, ssim, gpuSsim, std::abs(ssim - gpuSsim));
    }
    double maxGradient = 0.0;
    for (const float gradient : gradients)
        maxGradient = std::max(maxGradient, double{std::abs(gradient)});
    constexpr float gpuStep = 1e-2f;
    constexpr float cpuStep = 1e-4f;
    double gpuError = 0.0;
    double cpuError = 0.0;
    double gpuRelative = 0.0;
    double cpuRelative = 0.0;
    int checked = 0;
    for (int sample = 0; sample < 400 && checked < 120; ++sample) {
        const std::size_t index = random() % valueCount;
        const float original = values[index];
        if (std::abs(original - reference.getPixels()[index]) < 3.0f * gpuStep || original < gpuStep || original > 1.0f - gpuStep)
            continue;
        ++checked;
        values[index] = original + gpuStep;
        const std::expected<double, std::string> plus = bench.evaluate(values, nullptr);
        values[index] = original - gpuStep;
        const std::expected<double, std::string> minus = bench.evaluate(values, nullptr);
        values[index] = original + cpuStep;
        const double cpuSmallPlus = getCpuLoss(values, reference, kind);
        values[index] = original - cpuStep;
        const double cpuSmallMinus = getCpuLoss(values, reference, kind);
        values[index] = original;
        if (!plus || !minus)
            return std::unexpected(!plus ? plus.error() : minus.error());
        const double gpuDifference = (*plus - *minus) / (double{original + gpuStep} - double{original - gpuStep});
        const double cpuDifference = (cpuSmallPlus - cpuSmallMinus) / (double{original + cpuStep} - double{original - cpuStep});
        const double gradient = gradients[index];
        gpuError = std::max(gpuError, std::abs(gpuDifference - gradient));
        cpuError = std::max(cpuError, std::abs(cpuDifference - gradient));
        gpuRelative = std::max(gpuRelative, std::abs(gpuDifference - gradient) / std::abs(cpuDifference));
        cpuRelative = std::max(cpuRelative, std::abs(cpuDifference - gradient) / std::abs(cpuDifference));
    }
    std::println("{}: {} entries, largest |gradient| {:.3e}", name, checked, maxGradient);
    std::println("{}: against GPU finite differences (h = {}): max abs error {:.3e}, max relative error {:.3e}", name, gpuStep, gpuError, gpuRelative);
    std::println("{}: against CPU double finite differences (h = {}): max abs error {:.3e}, max relative error {:.3e}", name, cpuStep, cpuError, cpuRelative);
    if (std::abs(*value - cpuValue) > 1e-5 || cpuRelative > 1e-3 || gpuRelative > 2e-2)
        return std::unexpected(std::format("{} differs from the reference", name));
    return {};
}

}

Result testLoss(Device& device) {
    if (Result result = testKind(device, Training::LossKind::L1Ssim, "L1Ssim"); !result)
        return result;
    return testKind(device, Training::LossKind::L2, "L2");
}

}
