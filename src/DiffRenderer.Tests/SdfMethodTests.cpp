#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <numbers>
#include <print>
#include <random>
#include <string>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Methods/SdfGrid/SdfGrid.h"
#include "DiffRenderer/Training/Comparison.h"
#include "DiffRenderer/Training/ComparisonReport.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;

namespace {

constexpr std::uint32_t size = 128;
constexpr float focal = 140.0f;
constexpr float cameraDistance = 3.2f;
constexpr std::uint32_t trainCount = 48;
constexpr std::uint32_t testCount = 8;
constexpr std::uint32_t subsamples = 4;
constexpr int iterations = 3000;
constexpr double minimumPsnr = 35.0;
constexpr double minimumBoundaryGain = 10.0;

struct Vector {
    float x;
    float y;
    float z;
};

Vector operator+(const Vector a, const Vector b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vector operator-(const Vector a, const Vector b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vector operator*(const Vector a, const float b) {
    return {a.x * b, a.y * b, a.z * b};
}

float dot(const Vector a, const Vector b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector cross(const Vector a, const Vector b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vector normalize(const Vector a) {
    return a * (1.0f / std::sqrt(dot(a, a)));
}

float getDistance(const Vector point) {
    const float ring = std::hypot(point.x, point.y) - 0.7f;
    const float torus = std::hypot(ring, point.z) - 0.22f;
    const Vector offset = point - Vector{0.25f, 0.2f, 0.45f};
    const float sphere = std::sqrt(dot(offset, offset)) - 0.4f;
    return std::min(torus, sphere);
}

Vector getColour(const Vector point) {
    return {0.5f + 0.4f * std::sin(3.0f * point.x), 0.5f + 0.4f * std::sin(3.0f * point.y + 1.0f), 0.5f + 0.4f * std::sin(3.0f * point.z + 2.0f)};
}

Vector traceColour(const Vector origin, const Vector direction) {
    float distance = 0.0f;
    for (int step = 0; step < 512 && distance < 2.0f * cameraDistance; ++step) {
        const Vector point = origin + direction * distance;
        const float value = getDistance(point);
        if (value < 1e-5f)
            return getColour(point);
        distance += value;
    }
    return {1.0f, 1.0f, 1.0f};
}

Scenes::View makeView(const std::uint32_t index, const std::uint32_t count, const float phase) {
    const float height = 1.0f - 2.0f * (static_cast<float>(index) + 0.5f) / static_cast<float>(count);
    const float angle = static_cast<float>(index) * std::numbers::pi_v<float> * (3.0f - std::sqrt(5.0f)) + phase;
    const float radius = std::sqrt(1.0f - height * height);
    const Vector eye = Vector{radius * std::cos(angle), radius * std::sin(angle), height} * cameraDistance;
    const Vector forward = normalize(eye * -1.0f);
    const Vector up = std::abs(forward.z) > 0.99f ? Vector{1.0f, 0.0f, 0.0f} : Vector{0.0f, 0.0f, 1.0f};
    const Vector right = normalize(cross(forward, up));
    const Vector down = cross(forward, right);
    Scenes::View view;
    view.name = std::format("view{}", index);
    view.camera.worldToCamera = {
        {{right.x, right.y, right.z, -dot(right, eye)}, {down.x, down.y, down.z, -dot(down, eye)}, {forward.x, forward.y, forward.z, -dot(forward, eye)}}};
    view.camera.intrinsics = {focal, focal, 0.5f * size, 0.5f * size};
    view.camera.width = size;
    view.camera.height = size;
    view.image = Core::Image(size, size);
    std::span<float> pixels = view.image.getPixels();
    for (std::uint32_t y = 0; y < size; ++y)
        for (std::uint32_t x = 0; x < size; ++x) {
            Vector sum{0.0f, 0.0f, 0.0f};
            for (std::uint32_t sample = 0; sample < subsamples * subsamples; ++sample) {
                const float filmX = (static_cast<float>(x) + (static_cast<float>(sample % subsamples) + 0.5f) / subsamples - 0.5f * size) / focal;
                const float filmY = (static_cast<float>(y) + (static_cast<float>(sample / subsamples) + 0.5f) / subsamples - 0.5f * size) / focal;
                sum = sum + traceColour(eye, normalize(forward + right * filmX + down * filmY));
            }
            const Vector colour = sum * (1.0f / static_cast<float>(subsamples * subsamples));
            const std::size_t pixel = (std::size_t{y} * size + x) * 3;
            pixels[pixel] = colour.x;
            pixels[pixel + 1] = colour.y;
            pixels[pixel + 2] = colour.z;
        }
    return view;
}

Scenes::Dataset makeDataset() {
    Scenes::Dataset dataset;
    dataset.id = "sdf-shapes";
    dataset.name = "Torus and sphere";
    dataset.kind = Scenes::Kind::Object;
    dataset.background = {1.0f, 1.0f, 1.0f};
    for (std::uint32_t index = 0; index < trainCount; ++index)
        dataset.train.push_back(makeView(index, trainCount, 0.0f));
    for (std::uint32_t index = 0; index < testCount; ++index)
        dataset.test.push_back(makeView(index, testCount, 0.7f));
    std::mt19937 random(5);
    std::uniform_real_distribution<float> uniform(-1.3f, 1.3f);
    for (int point = 0; point < 1000; ++point)
        dataset.points.positions.push_back({uniform(random), uniform(random), uniform(random)});
    return dataset;
}

std::expected<double, std::string> train(Device& device, const Scenes::Dataset& dataset, const bool boundaryTerm) {
    Methods::SdfGrid method;
    const std::array<Methods::Method*, 1> methods{&method};
    const Training::Budget budget{.iterations = iterations, .seconds = 0.0, .evaluateEvery = 0, .testStride = 1};
    const Methods::Settings settings{.seed = 3, .iterations = iterations, .boundaryTerm = boundaryTerm};
    const std::filesystem::path output = std::format("/var/tmp/diffrenderer-sdf-test/{}", boundaryTerm ? "boundary" : "interior");
    Training::Comparison comparison;
    if (Result opened = check(comparison.open(device, dataset, methods, budget, settings, output)); !opened)
        return std::unexpected(opened.error());
    if (Result trained = check(comparison.runAll()); !trained)
        return std::unexpected(trained.error());
    const Training::MethodResult& result = comparison.getResults().front();
    if (Result written = check(Training::writeComparisonReport(comparison)); !written)
        return std::unexpected(written.error());
    std::println("boundary term {}: test PSNR {:.2f} dB, loss {:.4f} -> {:.4f}, {:.1f} s, renders in {}", boundaryTerm ? "on " : "off",
                 result.evaluations.back().psnr, result.losses.front(), result.losses.back(), result.trainSeconds, output.string());
    return result.evaluations.back().psnr;
}

}

//! Reconstructs an analytic torus and sphere from its views; the relaxed boundary term must reach a good PSNR and beat the interior term alone.
Result testSdfMethod(Device& device) {
    const Scenes::Dataset dataset = makeDataset();
    const std::expected<double, std::string> withBoundary = train(device, dataset, true);
    if (!withBoundary)
        return std::unexpected(withBoundary.error());
    const std::expected<double, std::string> withoutBoundary = train(device, dataset, false);
    if (!withoutBoundary)
        return std::unexpected(withoutBoundary.error());
    if (*withBoundary < minimumPsnr)
        return std::unexpected(std::format("the test PSNR {:.2f} dB is below {:.1f} dB", *withBoundary, minimumPsnr));
    if (*withBoundary < *withoutBoundary + minimumBoundaryGain)
        return std::unexpected(std::format("the boundary term gains only {:.2f} dB", *withBoundary - *withoutBoundary));
    return {};
}

}
