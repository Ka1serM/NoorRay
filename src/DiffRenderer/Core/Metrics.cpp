#include "DiffRenderer/Core/Metrics.h"

#include <array>
#include <cmath>
#include <vector>

namespace DiffRenderer::Core {

namespace {

constexpr int radius = 5;
constexpr std::size_t momentCount = 5;
constexpr double c1 = 0.01 * 0.01;
constexpr double c2 = 0.03 * 0.03;

std::array<double, 2 * radius + 1> getWindow() {
    std::array<double, 2 * radius + 1> window{};
    double sum = 0.0;
    for (int tap = -radius; tap <= radius; ++tap) {
        window[tap + radius] = std::exp(-tap * tap / (2.0 * 1.5 * 1.5));
        sum += window[tap + radius];
    }
    for (double& weight : window)
        weight /= sum;
    return window;
}

using Moments = std::array<double, momentCount>;

Moments getMoments(const double value, const double target) {
    return {value, target, value * value, target * target, value * target};
}

double getChannelSsim(const Image& image, const Image& reference, const std::size_t channel) {
    static const std::array<double, 2 * radius + 1> window = getWindow();
    const int width = static_cast<int>(image.getWidth());
    const int height = static_cast<int>(image.getHeight());
    const std::span<const float> values = image.getPixels();
    const std::span<const float> targets = reference.getPixels();
    std::vector<Moments> rows(std::size_t(width) * height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            Moments sums{};
            for (int tap = -radius; tap <= radius; ++tap) {
                if (x + tap < 0 || x + tap >= width)
                    continue;
                const std::size_t index = (std::size_t(y) * width + x + tap) * 3 + channel;
                const Moments moments = getMoments(values[index], targets[index]);
                for (std::size_t moment = 0; moment < momentCount; ++moment)
                    sums[moment] += window[tap + radius] * moments[moment];
            }
            rows[std::size_t(y) * width + x] = sums;
        }
    double total = 0.0;
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            Moments sums{};
            for (int tap = -radius; tap <= radius; ++tap) {
                if (y + tap < 0 || y + tap >= height)
                    continue;
                for (std::size_t moment = 0; moment < momentCount; ++moment)
                    sums[moment] += window[tap + radius] * rows[std::size_t(y + tap) * width + x][moment];
            }
            const double mean = sums[0];
            const double targetMean = sums[1];
            const double covariance = sums[4] - mean * targetMean;
            const double variances = sums[2] - mean * mean + sums[3] - targetMean * targetMean;
            total += (2.0 * mean * targetMean + c1) * (2.0 * covariance + c2) / ((mean * mean + targetMean * targetMean + c1) * (variances + c2));
        }
    return total;
}

}

double getSsim(const Image& image, const Image& reference) {
    if (image.isEmpty())
        return 1.0;
    double total = 0.0;
    for (std::size_t channel = 0; channel < 3; ++channel)
        total += getChannelSsim(image, reference, channel);
    return total / static_cast<double>(image.getPixels().size());
}

}
