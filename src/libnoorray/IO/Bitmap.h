#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <glm/vec4.hpp>

class Bitmap
{
public:
    static_assert(sizeof(glm::vec4) == sizeof(float) * 4);

    Bitmap(const uint32_t width, const uint32_t height, std::vector<glm::vec4> pixels)
        : width_(width), height_(height), pixels_(std::move(pixels))
    {
        if (width_ == 0 || height_ == 0)
            throw std::invalid_argument("Bitmap dimensions must be greater than zero");
        if (pixels_.size() != static_cast<size_t>(width_) * height_)
            throw std::invalid_argument("Bitmap pixel count does not match its dimensions");
    }

    uint32_t width() const { return width_; }
    uint32_t height() const { return height_; }
    const std::vector<glm::vec4>& pixels() const { return pixels_; }
    const glm::vec4& pixel(const uint32_t x, const uint32_t y) const
    {
        if (x >= width_ || y >= height_)
            throw std::out_of_range("Bitmap pixel coordinate is outside the image");
        return pixels_[static_cast<size_t>(y) * width_ + x];
    }
    const float* rgba() const { return &pixels_.front().x; }

private:
    uint32_t width_{};
    uint32_t height_{};
    std::vector<glm::vec4> pixels_;
};
