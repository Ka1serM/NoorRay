#include "DiffRenderer/Algorithms/VolumeRendering/Phantom.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace DiffRenderer::Algorithms {

namespace {

struct Point {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

constexpr Point tumorCenter{0.13, 0.1, -0.06};
constexpr double tumorRadius = 0.085;

double hash(const std::int32_t x, const std::int32_t y, const std::int32_t z) {
    std::uint32_t value = static_cast<std::uint32_t>(x) * 73856093u ^ static_cast<std::uint32_t>(y) * 19349663u ^ static_cast<std::uint32_t>(z) * 83492791u;
    value ^= value >> 13;
    value *= 0x5bd1e995u;
    value ^= value >> 15;
    return static_cast<double>(value & 0xffffffu) / static_cast<double>(0xffffffu) * 2.0 - 1.0;
}

double fade(const double t) {
    return t * t * (3.0 - 2.0 * t);
}

double mix(const double a, const double b, const double t) {
    return a + (b - a) * t;
}

double valueNoise(const Point p) {
    const double fx = std::floor(p.x);
    const double fy = std::floor(p.y);
    const double fz = std::floor(p.z);
    const auto x = static_cast<std::int32_t>(fx);
    const auto y = static_cast<std::int32_t>(fy);
    const auto z = static_cast<std::int32_t>(fz);
    const double u = fade(p.x - fx);
    const double v = fade(p.y - fy);
    const double w = fade(p.z - fz);
    const double x00 = mix(hash(x, y, z), hash(x + 1, y, z), u);
    const double x10 = mix(hash(x, y + 1, z), hash(x + 1, y + 1, z), u);
    const double x01 = mix(hash(x, y, z + 1), hash(x + 1, y, z + 1), u);
    const double x11 = mix(hash(x, y + 1, z + 1), hash(x + 1, y + 1, z + 1), u);
    return mix(mix(x00, x10, v), mix(x01, x11, v), w);
}

double fractalNoise(const Point p, const double frequency) {
    double sum = 0.0;
    double amplitude = 0.5;
    double scale = frequency;
    for (int octave = 0; octave < 4; ++octave) {
        sum += amplitude * valueNoise({p.x * scale + 17.0 * octave, p.y * scale, p.z * scale});
        amplitude *= 0.5;
        scale *= 2.0;
    }
    return sum;
}

double ellipsoid(const Point p, const Point center, const Point radii) {
    const double x = (p.x - center.x) / radii.x;
    const double y = (p.y - center.y) / radii.y;
    const double z = (p.z - center.z) / radii.z;
    return std::sqrt(x * x + y * y + z * z);
}

double inside(const double radius, const double edge, const double width) {
    return 1.0 - std::clamp((radius - edge) / width + 0.5, 0.0, 1.0);
}

}

Phantom::Phantom(const std::uint32_t size) : size(size), density(std::size_t{size} * size * size), tumor(density.size()) {
    const double voxel = 1.0 / size;
    const double soft = 2.0 * voxel;
    for (std::uint32_t z = 0; z < size; ++z)
        for (std::uint32_t y = 0; y < size; ++y)
            for (std::uint32_t x = 0; x < size; ++x) {
                const Point p{(x + 0.5) * voxel - 0.5, (y + 0.5) * voxel - 0.5, (z + 0.5) * voxel - 0.5};
                const double head = ellipsoid(p, {0.0, -0.02, 0.0}, {0.4, 0.44, 0.38}) + 0.03 * fractalNoise(p, 6.0);
                const double cortex = fractalNoise(p, 14.0);
                double value = 0.22 * inside(head, 1.0, soft / 0.4);
                value = mix(value, 0.95, inside(head, 0.95, soft / 0.4) - inside(head, 0.88, soft / 0.4));
                value = mix(value, 0.45 + 0.1 * cortex, inside(head, 0.86, soft / 0.4));
                const double left = ellipsoid(p, {-0.05, 0.04, 0.0}, {0.03, 0.07, 0.13});
                const double right = ellipsoid(p, {0.05, 0.04, 0.0}, {0.03, 0.07, 0.13});
                value = mix(value, 0.12, inside(std::min(left, right), 1.0, soft / 0.05));
                const double lesion = ellipsoid(p, tumorCenter, {tumorRadius, tumorRadius, tumorRadius}) + 0.25 * fractalNoise(p, 18.0);
                value = mix(value, 0.53, inside(lesion, 1.7, soft / tumorRadius));
                const double mass = inside(lesion, 1.0, soft / tumorRadius);
                value = mix(value, 0.74, mass);
                value = mix(value, 0.63, inside(lesion, 0.55, soft / tumorRadius));
                const std::size_t index = (std::size_t{z} * size + y) * size + x;
                density[index] = static_cast<float>(std::clamp(value, 0.0, 1.0));
                tumor[index] = static_cast<float>(mass);
            }
}

}
