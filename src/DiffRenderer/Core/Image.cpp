#include "DiffRenderer/Core/Image.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace DiffRenderer::Core {

namespace {

std::uint32_t toByte(const float encoded) {
    return static_cast<std::uint32_t>(std::clamp(encoded, 0.0f, 1.0f) * 255.0f + 0.5f);
}

std::uint32_t toSrgbByte(const float linear) {
    const float clamped = std::clamp(linear, 0.0f, 1.0f);
    return toByte(clamped <= 0.0031308f ? 12.92f * clamped : 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f);
}

template <auto Encode>
void pack(const std::span<const float> values, const std::span<std::uint32_t> rgba) {
    for (std::size_t pixel = 0; pixel < rgba.size(); ++pixel)
        rgba[pixel] = Encode(values[pixel * 3]) | Encode(values[pixel * 3 + 1]) << 8 | Encode(values[pixel * 3 + 2]) << 16 | 0xff000000u;
}

}

double getMeanSquaredError(const Image& image, const Image& reference) {
    const std::span<const float> values = image.getPixels();
    const std::span<const float> expected = reference.getPixels();
    double sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const double difference = double{values[index]} - expected[index];
        sum += difference * difference;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

double getPsnr(const double meanSquaredError) {
    if (meanSquaredError <= 0.0)
        return std::numeric_limits<double>::infinity();
    return -10.0 * std::log10(meanSquaredError);
}

double getL2Loss(const Image& image, const Image& reference, const std::span<float> gradient) {
    const std::span<const float> values = image.getPixels();
    const std::span<const float> expected = reference.getPixels();
    const float scale = 2.0f / static_cast<float>(values.size());
    double sum = 0.0;
    for (std::size_t index = 0; index < values.size(); ++index) {
        const float difference = values[index] - expected[index];
        sum += double{difference} * difference;
        gradient[index] = scale * difference;
    }
    return sum / static_cast<double>(values.size());
}

void toRgba8(const Image& image, const std::span<std::uint32_t> rgba) {
    if (image.getEncoding() == Image::Encoding::Srgb)
        return pack<toByte>(image.getPixels(), rgba);
    pack<toSrgbByte>(image.getPixels(), rgba);
}

void getDifference(const Image& image, const Image& reference, const float scale, Image& difference) {
    if (difference.getWidth() != image.getWidth() || difference.getHeight() != image.getHeight())
        difference = Image(image.getWidth(), image.getHeight());
    const std::span<const float> values = image.getPixels();
    const std::span<const float> expected = reference.getPixels();
    const std::span<float> result = difference.getPixels();
    for (std::size_t pixel = 0; pixel * 3 < values.size(); ++pixel) {
        float sum = 0.0f;
        for (std::size_t channel = 0; channel < 3; ++channel)
            sum += std::abs(values[pixel * 3 + channel] - expected[pixel * 3 + channel]);
        const float shown = std::min(sum * scale / 3.0f, 1.0f);
        result[pixel * 3] = shown;
        result[pixel * 3 + 1] = shown * shown;
        result[pixel * 3 + 2] = 0.2f * (1.0f - shown) * shown;
    }
}

}
