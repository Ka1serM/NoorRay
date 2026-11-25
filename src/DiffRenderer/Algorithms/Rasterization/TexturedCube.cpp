#include "DiffRenderer/Algorithms/Rasterization/TexturedCube.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace DiffRenderer::Algorithms {

namespace {

using Matrix = std::array<float, 16>;

constexpr float perturbation = 0.12f;
constexpr Matrix view{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -3.0f, 0, 0, 0, 1};
constexpr std::array<std::array<std::uint32_t, 4>, 6> faces{{{1, 3, 7, 5}, {0, 4, 6, 2}, {2, 6, 7, 3}, {0, 1, 5, 4}, {4, 5, 7, 6}, {0, 2, 3, 1}}};

Matrix multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            for (int index = 0; index < 4; ++index)
                result[row * 4 + column] += a[row * 4 + index] * b[index * 4 + column];
    return result;
}

//! Clip space y points down, so that the first image row is the top of the view.
Matrix getProjection(const float fovY, const float near, const float far) {
    const float focal = 1.0f / std::tan(fovY * 0.5f);
    return {focal, 0, 0, 0, 0, -focal, 0, 0, 0, 0, (near + far) / (near - far), 2 * near * far / (near - far), 0, 0, -1, 0};
}

Matrix getRotationX(const float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1};
}

Matrix getRotationY(const float angle) {
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    return {c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1};
}

}

TexturedCube::TexturedCube() {
    for (std::uint32_t corner = 0; corner < 8; ++corner)
        positions.insert(positions.end(), {(corner & 1) ? 0.5f : -0.5f, (corner & 2) ? 0.5f : -0.5f, (corner & 4) ? 0.5f : -0.5f});
    for (std::uint32_t face = 0; face < faces.size(); ++face) {
        const std::array<std::uint32_t, 4>& quad = faces[face];
        triangles.insert(triangles.end(), {quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]});
        const float u = static_cast<float>(face % 3) / 3.0f;
        const float v = static_cast<float>(face / 3) / 2.0f;
        const auto first = static_cast<std::uint32_t>(uvs.size());
        uvs.insert(uvs.end(), {{u, v}, {u + 1.0f / 3.0f, v}, {u + 1.0f / 3.0f, v + 0.5f}, {u, v + 0.5f}});
        uvTriangles.insert(uvTriangles.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
    }
    std::map<std::pair<std::uint32_t, std::uint32_t>, std::vector<std::uint32_t>> edges;
    for (std::uint32_t corner = 0; corner < triangles.size(); ++corner) {
        const std::uint32_t base = corner / 3 * 3;
        edges[std::minmax(triangles[base + (corner + 1) % 3], triangles[base + (corner + 2) % 3])].push_back(triangles[corner]);
    }
    for (std::uint32_t corner = 0; corner < triangles.size(); ++corner) {
        const std::uint32_t base = corner / 3 * 3;
        std::int32_t opposite = -1;
        for (const std::uint32_t apex : edges[std::minmax(triangles[base + (corner + 1) % 3], triangles[base + (corner + 2) % 3])])
            if (apex != triangles[corner])
                opposite = static_cast<std::int32_t>(apex);
        opposites.push_back(opposite);
    }
    texture.resize(std::size_t{textureWidth} * textureHeight * 3);
    for (std::uint32_t y = 0; y < textureHeight; ++y)
        for (std::uint32_t x = 0; x < textureWidth; ++x)
            for (std::uint32_t channel = 0; channel < 3; ++channel) {
                const float wave = std::sin(0.35f * x + 2.1f * channel) * std::cos(0.27f * y - channel);
                const float checker = ((x / 8 + y / 8) & 1) ? 0.12f : -0.12f;
                texture[(y * textureWidth + x) * 3 + channel] = 0.5f + 0.3f * wave + checker;
            }
    transform = multiply(getProjection(0.8f, 0.5f, 10.0f), multiply(view, multiply(getRotationX(0.45f), getRotationY(0.7f))));
}

void TexturedCube::project(const std::span<const float> points, const std::span<NoorRHI::float4> clip) const {
    for (std::size_t vertex = 0; vertex < clip.size(); ++vertex) {
        const std::array<float, 4> point{points[vertex * 3], points[vertex * 3 + 1], points[vertex * 3 + 2], 1.0f};
        std::array<float, 4> result{};
        for (int row = 0; row < 4; ++row)
            for (int column = 0; column < 4; ++column)
                result[row] += transform[row * 4 + column] * point[column];
        clip[vertex] = {result[0], result[1], result[2], result[3]};
    }
}

void TexturedCube::pullBack(const std::span<const NoorRHI::float4> clipGradient, const std::span<float> gradient) const {
    for (std::size_t vertex = 0; vertex < clipGradient.size(); ++vertex) {
        const NoorRHI::float4& value = clipGradient[vertex];
        const std::array<float, 4> clip{value.x, value.y, value.z, value.w};
        for (int column = 0; column < 3; ++column) {
            float sum = 0.0f;
            for (int row = 0; row < 4; ++row)
                sum += transform[row * 4 + column] * clip[row];
            gradient[vertex * 3 + column] = sum;
        }
    }
}

std::vector<float> TexturedCube::perturb(Core::Random& random) const {
    std::vector<float> start = positions;
    for (float& value : start)
        value += random.uniform(-perturbation, perturbation);
    return start;
}

}
