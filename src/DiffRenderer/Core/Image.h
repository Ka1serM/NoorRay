#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace DiffRenderer::Core {

//! RGB, three floats for each pixel, rows from the top.
class Image {
public:
    //! Srgb keeps the values as an image file stores them, divided by 255.
    enum class Encoding : std::uint8_t { Linear, Srgb };

    Image() = default;
    Image(std::uint32_t width, std::uint32_t height) : width(width), height(height), pixels(std::size_t{width} * height * 3) {}

    std::uint32_t getWidth() const { return width; }
    std::uint32_t getHeight() const { return height; }
    std::span<float> getPixels() { return pixels; }
    std::span<const float> getPixels() const { return pixels; }
    bool isEmpty() const { return pixels.empty(); }
    Encoding getEncoding() const { return encoding; }
    void setEncoding(const Encoding value) { encoding = value; }

private:
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::vector<float> pixels;
    Encoding encoding = Encoding::Linear;
};

double getMeanSquaredError(const Image& image, const Image& reference);
double getPsnr(double meanSquaredError);
//! The L2 loss of the image against the reference, the mean of the squared differences; fills the gradient of the loss with respect to each value.
double getL2Loss(const Image& image, const Image& reference, std::span<float> gradient);
//! Packs the image as RGBA8 for the display, with an sRGB transfer for a linear image.
void toRgba8(const Image& image, std::span<std::uint32_t> rgba);
//! The absolute difference of the two images, scaled so that a difference of 1/scale is white.
void getDifference(const Image& image, const Image& reference, float scale, Image& difference);

}
