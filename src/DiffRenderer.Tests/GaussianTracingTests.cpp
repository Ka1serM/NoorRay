#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <limits>
#include <numbers>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "DiffRenderer.Tests/Tests.h"
#include "DiffRenderer/Algorithms/GaussianTracing/GaussianTracer.h"
#include "DiffRenderer/Core/GaussianProxies.h"
#include "NoorRay/Spectral/SpectralTables.h"

namespace DiffRenderer::Tests {

using namespace NoorRHI;
using namespace Algorithms;

namespace {

constexpr std::array<std::string_view, 5> groupNames{"means", "quaternions", "scales", "opacities", "colors"};

struct Vector {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    static Vector from(const float3 value) { return {value.x, value.y, value.z}; }
};

Vector operator+(const Vector a, const Vector b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

Vector operator-(const Vector a, const Vector b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vector operator*(const Vector a, const double factor) {
    return {a.x * factor, a.y * factor, a.z * factor};
}

double dot(const Vector a, const Vector b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vector cross(const Vector a, const Vector b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Vector normalize(const Vector value) {
    return value * (1.0 / std::sqrt(dot(value, value)));
}

double toLinear(const double encoded) {
    return encoded <= 0.04045 ? encoded / 12.92 : std::pow((encoded + 0.055) / 1.055, 2.4);
}

struct Contribution {
    double t = 0.0;
    std::size_t gaussian = 0;
    double alpha = 0.0;
};

//! Evaluates every Gaussian for every ray in double precision and sorts the list.
class ReferenceRenderer {
public:
    ReferenceRenderer(const Gaussians& gaussians, const GaussianScene& scene) : gaussians(gaussians), scene(scene) {
        const Core::GaussianProxyMesh mesh;
        for (std::size_t first = 0; first < mesh.getIndices().size(); first += 3) {
            const Vector a = getVertex(mesh, first);
            const Vector normal = normalize(cross(getVertex(mesh, first + 1) - a, getVertex(mesh, first + 2) - a));
            planes.push_back({normal, dot(normal, a)});
        }
    }

    std::vector<float> render(const std::uint32_t size) const {
        const Vector origin = Vector::from(scene.cameraOrigin);
        const Vector forward = normalize(Vector::from(scene.cameraTarget) - origin);
        const Vector right = normalize(cross(forward, Vector::from(scene.cameraUp)));
        const Vector up = cross(right, forward);
        const double tangent = std::tan(scene.fieldOfView * std::numbers::pi / 360.0);
        std::vector<float> image;
        for (std::uint32_t y = 0; y < size; ++y)
            for (std::uint32_t x = 0; x < size; ++x) {
                const double u = (2.0 * x + 1.0) / size - 1.0;
                const double v = 1.0 - (2.0 * y + 1.0) / size;
                for (const double value : getRadiance(origin, normalize(forward + right * (u * tangent) + up * (v * tangent))))
                    image.push_back(static_cast<float>(value));
            }
        return image;
    }

private:
    struct Plane {
        Vector normal;
        double distance = 0.0;
    };

    static Vector getVertex(const Core::GaussianProxyMesh& mesh, const std::size_t index) { return Vector::from(mesh.getVertices()[mesh.getIndices()[index]]); }

    //! The coordinates on the axes of the Gaussian, in standard deviations.
    Vector toLocal(const std::size_t gaussian, const Vector value) const {
        const float* const q = &gaussians.quaternions[4 * gaussian];
        const float* const scale = &gaussians.scales[3 * gaussian];
        const double length = std::sqrt(double{q[0]} * q[0] + double{q[1]} * q[1] + double{q[2]} * q[2] + double{q[3]} * q[3]);
        const Vector axis = Vector{q[1], q[2], q[3]} * (-1.0 / length);
        const Vector t = cross(axis, value) * 2.0;
        const Vector turned = value + t * (q[0] / length) + cross(axis, t);
        return {turned.x / scale[0], turned.y / scale[1], turned.z / scale[2]};
    }

    bool entersProxy(const Vector origin, const Vector direction, const double cutoff) const {
        double entry = -std::numeric_limits<double>::infinity();
        double exit = std::numeric_limits<double>::infinity();
        for (const Plane& plane : planes) {
            const double slope = dot(plane.normal, direction);
            const double t = (plane.distance * cutoff - dot(plane.normal, origin)) / slope;
            if (slope < 0.0)
                entry = std::max(entry, t);
            else
                exit = std::min(exit, t);
        }
        return entry >= 0.0 && entry <= exit;
    }

    double getMeshDistance(const Vector origin, const Vector direction) const {
        double nearest = std::numeric_limits<double>::infinity();
        for (std::size_t first = 0; first < scene.indices.size(); first += 3) {
            const Vector a = Vector::from(scene.positions[scene.indices[first]]);
            const Vector edge1 = Vector::from(scene.positions[scene.indices[first + 1]]) - a;
            const Vector edge2 = Vector::from(scene.positions[scene.indices[first + 2]]) - a;
            const Vector p = cross(direction, edge2);
            const double scale = 1.0 / dot(edge1, p);
            const Vector offset = origin - a;
            const Vector q = cross(offset, edge1);
            const double u = dot(offset, p) * scale;
            const double v = dot(direction, q) * scale;
            const double t = dot(edge2, q) * scale;
            if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 0.0)
                nearest = std::min(nearest, t);
        }
        return nearest;
    }

    std::optional<Contribution> getContribution(const std::size_t gaussian, const Vector origin, const Vector direction, const double tMax) const {
        const double opacity = gaussians.opacities[gaussian];
        if (opacity * 255.0 <= 1.0)
            return {};
        const Vector mean{gaussians.means[3 * gaussian], gaussians.means[3 * gaussian + 1], gaussians.means[3 * gaussian + 2]};
        const Vector localOrigin = toLocal(gaussian, origin - mean);
        const Vector localDirection = toLocal(gaussian, direction);
        if (!entersProxy(localOrigin, localDirection, std::sqrt(2.0 * std::log(opacity * 255.0))))
            return {};
        const double t = -dot(localOrigin, localDirection) / dot(localDirection, localDirection);
        if (t <= 0.0 || t >= tMax)
            return {};
        const Vector closest = localOrigin + localDirection * t;
        const double alpha = std::min(opacity * std::exp(-0.5 * dot(closest, closest)), 0.999);
        if (alpha < 1.0 / 255.0)
            return {};
        return Contribution{t, gaussian, alpha};
    }

    std::array<double, 3> getRadiance(const Vector origin, const Vector direction) const {
        const double tMax = getMeshDistance(origin, direction);
        std::vector<Contribution> list;
        for (std::size_t gaussian = 0; gaussian < gaussians.getCount(); ++gaussian)
            if (const std::optional<Contribution> contribution = getContribution(gaussian, origin, direction, tMax))
                list.push_back(*contribution);
        std::ranges::sort(list, {}, [](const Contribution& entry) { return std::pair(entry.t, entry.gaussian); });
        std::array<double, 3> radiance{};
        double transmittance = 1.0;
        for (const Contribution& entry : list) {
            for (std::size_t channel = 0; channel < 3; ++channel)
                radiance[channel] += transmittance * entry.alpha * toLinear(std::max(double{gaussians.colors[3 * entry.gaussian + channel]}, 0.0));
            transmittance *= 1.0 - entry.alpha;
        }
        //! The light of the scene is off, so the mesh is black.
        if (!std::isinf(tMax))
            return radiance;
        radiance[0] += transmittance * scene.environment.x;
        radiance[1] += transmittance * scene.environment.y;
        radiance[2] += transmittance * scene.environment.z;
        return radiance;
    }

    const Gaussians& gaussians;
    const GaussianScene& scene;
    std::vector<Plane> planes;
};

Result testForward(Device& device) {
    constexpr std::uint32_t size = 128;
    GaussianScene scene = getFloorAndBox();
    scene.lightIntensity = 0.0f;
    Core::Random random(11);
    const Gaussians gaussians = makeRandomGaussians(300, random);
    GaussianTracer tracer;
    std::vector<float> image(std::size_t{size} * size * 3);
    for (std::expected<void, Error> step : {tracer.open(device, scene, gaussians.getCount(), size, size), tracer.setParameters(gaussians),
                                            tracer.render(GaussianShading::Emissive, {.samples = 1, .spectral = false, .filterWidth = 0.0f}, image)})
        if (!step)
            return check(step);
    const std::vector<float> reference = ReferenceRenderer(gaussians, scene).render(size);
    double sum = 0.0;
    double largest = 0.0;
    for (std::size_t index = 0; index < image.size(); ++index) {
        const double error = std::abs(double{image[index]} - reference[index]);
        sum += error;
        largest = std::max(largest, error);
    }
    const double mean = sum / static_cast<double>(image.size());
    std::println("forward: mean error {:.2e}, largest error {:.2e}", mean, largest);
    //! A ray that grazes the cutoff gains or loses a Gaussian of at most 1/255.
    if (!(mean < 1e-5 && largest < 5e-3))
        return std::unexpected("the composite differs from the sorted reference");
    return {};
}

//! Thin discs in front of each other. They reach out of the view, and no path vertex enters another disc, so the image has no discontinuity that moves.
Gaussians makeLayers() {
    constexpr std::size_t layers = 5;
    Core::Random random(5);
    Gaussians gaussians(layers);
    for (std::size_t layer = 0; layer < layers; ++layer) {
        gaussians.means[3 * layer] = random.uniform(-0.2f, 0.2f);
        gaussians.means[3 * layer + 1] = random.uniform(-0.2f, 0.2f);
        gaussians.means[3 * layer + 2] = 0.8f - 0.4f * static_cast<float>(layer);
        gaussians.quaternions[4 * layer] = 1.0f;
        gaussians.quaternions[4 * layer + 1] = random.uniform(-0.015f, 0.015f);
        gaussians.quaternions[4 * layer + 2] = random.uniform(-0.015f, 0.015f);
        gaussians.quaternions[4 * layer + 3] = random.uniform(-0.6f, 0.6f);
        gaussians.scales[3 * layer] = random.uniform(0.6f, 0.65f);
        gaussians.scales[3 * layer + 1] = random.uniform(0.7f, 0.8f);
        gaussians.scales[3 * layer + 2] = 0.03f;
        gaussians.opacities[layer] = random.uniform(0.4f, 0.65f);
        for (std::size_t channel = 0; channel < 3; ++channel)
            gaussians.colors[3 * layer + channel] = random.uniform(0.4f, 0.9f);
    }
    return gaussians;
}

GaussianScene makeWall() {
    GaussianScene scene;
    scene.cameraOrigin = {0.0f, 0.0f, 5.0f};
    scene.cameraTarget = {};
    scene.fieldOfView = 20.0f;
    scene.lightPosition = {0.7f, 1.1f, 3.5f};
    scene.addQuad({-200.0f, -200.0f, -1.6f}, {400.0f, 0.0f, 0.0f}, {0.0f, 400.0f, 0.0f}, {0.6f, 0.5f, 0.4f});
    return scene;
}

struct Estimate {
    void add(const double value) {
        ++count;
        const double delta = value - mean;
        mean += delta / count;
        squares += delta * (value - mean);
    }

    double getError() const { return std::sqrt(squares / (count * (count - 1.0))); }

    int count = 0;
    double mean = 0.0;
    double squares = 0.0;
};

struct GradientTest {
    std::string_view name;
    GaussianShading shading = GaussianShading::Emissive;
    std::uint32_t samples = 0;
    float step = 0.0f;
    //! Relative to the directional derivative; the standard error must be below half of it.
    double tolerance = 0.0;
};

class GradientCheck {
public:
    Result open(Device& device, const GaussianScene& scene) {
        weights.resize(std::size_t{size} * size * 3);
        for (std::size_t index = 0; index < weights.size(); ++index)
            weights[index] = (1.0f + 0.5f * std::sin(static_cast<float>(index) * 0.021f)) / static_cast<float>(weights.size());
        image.resize(weights.size());
        return check(tracer.open(device, scene, gaussians.getCount(), size, size));
    }

    Result run(const GradientTest& test) {
        std::string failures;
        Gaussians gradient(gaussians.getCount());
        if (Result derived = differentiate(test, 1, gradient); !derived)
            return derived;
        for (std::size_t group = 0; group < groupNames.size(); ++group) {
            const std::expected<bool, std::string> passed = compare(test, group, getDirection(*gradient.getGroups()[group]));
            if (!passed)
                return std::unexpected(passed.error());
            if (!*passed)
                failures += std::format("{} {}; ", test.name, groupNames[group]);
        }
        if (!failures.empty())
            return std::unexpected("the gradient differs from finite differences: " + failures);
        return {};
    }

private:
    static constexpr std::uint32_t size = 64;
    static constexpr int minBatches = 4;
    static constexpr int maxBatches = 48;

    //! The gradient itself, with a random factor for each component, so that errors of different components do not cancel.
    std::vector<float> getDirection(const std::vector<float>& gradient) {
        std::vector<float> direction = gradient;
        double squares = 0.0;
        for (float& value : direction) {
            value *= random.uniform(0.4f, 1.6f);
            squares += double{value} * value;
        }
        const float factor = static_cast<float>(1.0 / std::sqrt(squares));
        for (float& value : direction)
            value *= factor;
        return direction;
    }

    Result differentiate(const GradientTest& test, const std::uint32_t seed, Gaussians& result) {
        if (Result uploaded = check(tracer.setParameters(gaussians)); !uploaded)
            return uploaded;
        return check(tracer.backward(test.shading, {.samples = test.samples, .seed = seed}, weights, result));
    }

    std::expected<double, std::string> getLoss(const GradientTest& test, const std::uint32_t seed, const Gaussians& values) {
        for (std::expected<void, Error> step : {tracer.setParameters(values), tracer.render(test.shading, {.samples = test.samples, .seed = seed}, image)})
            if (!step)
                return std::unexpected(step.error().message);
        double loss = 0.0;
        for (std::size_t index = 0; index < image.size(); ++index)
            loss += double{image[index]} * weights[index];
        return loss;
    }

    std::expected<double, std::string> getNumeric(const GradientTest& test, const std::uint32_t seed, const std::size_t group,
                                                  const std::span<const float> direction) {
        std::array<double, 2> losses{};
        for (std::size_t side = 0; side < losses.size(); ++side) {
            Gaussians moved = gaussians;
            const float distance = side == 0 ? -test.step : test.step;
            for (std::size_t index = 0; index < direction.size(); ++index)
                (*moved.getGroups()[group])[index] += distance * direction[index];
            const std::expected<double, std::string> loss = getLoss(test, seed, moved);
            if (!loss)
                return loss;
            losses[side] = *loss;
        }
        return (losses[1] - losses[0]) / (2.0 * test.step);
    }

    std::expected<bool, std::string> compare(const GradientTest& test, const std::size_t group, const std::vector<float>& direction) {
        Estimate analytic;
        Estimate numeric;
        Estimate difference;
        Gaussians gradient(gaussians.getCount());
        for (int batch = 0; batch < maxBatches; ++batch) {
            const auto seed = static_cast<std::uint32_t>(100 + 2 * batch);
            if (Result derived = differentiate(test, seed + 1, gradient); !derived)
                return std::unexpected(derived.error());
            double slope = 0.0;
            for (std::size_t index = 0; index < direction.size(); ++index)
                slope += double{direction[index]} * (*gradient.getGroups()[group])[index];
            const std::expected<double, std::string> measured = getNumeric(test, seed, group, direction);
            if (!measured)
                return std::unexpected(measured.error());
            analytic.add(slope);
            numeric.add(*measured);
            difference.add(slope - *measured);
            if (difference.count >= minBatches && difference.getError() < 0.15 * test.tolerance * std::abs(numeric.mean))
                break;
        }
        const double allowed = test.tolerance * std::abs(numeric.mean);
        std::println("{:<10} {:<11} analytic {:+.5e}, finite difference {:+.5e}, difference {:+.1e} +- {:.1e}, {} batches", test.name, groupNames[group],
                     analytic.mean, numeric.mean, difference.mean, difference.getError(), difference.count);
        return difference.getError() < 0.5 * allowed && std::abs(difference.mean) <= allowed + 3.0 * difference.getError();
    }

    Gaussians gaussians = makeLayers();
    Core::Random random{23};
    GaussianTracer tracer;
    std::vector<float> weights;
    std::vector<float> image;
};

Result testGradients(Device& device) {
    GaussianScene emissive = makeWall();
    emissive.addBox({0.1f, -0.5f, 1.1f}, {0.5f, -0.1f, 1.3f}, {0.2f, 0.45f, 0.65f});
    const std::array<std::pair<GaussianScene, GradientTest>, 2> tests{
        {{emissive, {.name = "emissive", .shading = GaussianShading::Emissive, .samples = 64, .step = 0.005f, .tolerance = 0.01}},
         {makeWall(), {.name = "scattering", .shading = GaussianShading::Scattering, .samples = 2048, .step = 0.015f, .tolerance = 0.02}}}};
    std::string failures;
    for (const auto& [scene, test] : tests) {
        GradientCheck gradientCheck;
        if (Result opened = gradientCheck.open(device, scene); !opened)
            return opened;
        if (const Result result = gradientCheck.run(test); !result)
            failures += result.error();
    }
    if (!failures.empty())
        return std::unexpected(failures);
    return {};
}

Result testTable() {
    const auto entries = NoorRay::Spectral::SpectralTables::buildEntries();
    std::array<double, 9> sums{};
    for (const auto& entry : entries) {
        const std::array<double, 3> film{entry.filmAndPdf.x, entry.filmAndPdf.y, entry.filmAndPdf.z};
        const std::array<double, 3> basis{entry.basisAndLambda.x, entry.basisAndLambda.y, entry.basisAndLambda.z};
        for (std::size_t index = 0; index < sums.size(); ++index)
            sums[index] += film[index / 3] * basis[index % 3] / static_cast<double>(entries.size());
    }
    double largest = 0.0;
    for (std::size_t index = 0; index < sums.size(); ++index)
        largest = std::max(largest, std::abs(sums[index] - (index % 4 == 0 ? 1.0 : 0.0)));
    std::println("spectral table: largest error of the expected round trip {:.1e}", largest);
    if (!(largest < 1e-3))
        return std::unexpected("the spectral round trip changes the colour");
    return {};
}

//! The light is off: it has a blackbody spectrum, which the RGB mode samples at three wavelengths only.
Result testSpectralMean(Device& device) {
    constexpr std::uint32_t size = 64;
    GaussianScene scene = getFloorAndBox();
    scene.lightIntensity = 0.0f;
    Core::Random random(11);
    const Gaussians gaussians = makeRandomGaussians(300, random);
    GaussianTracer tracer;
    std::array<std::vector<float>, 2> images;
    for (std::expected<void, Error> step : {tracer.open(device, scene, gaussians.getCount(), size, size), tracer.setParameters(gaussians)})
        if (!step)
            return check(step);
    for (std::size_t mode = 0; mode < images.size(); ++mode) {
        images[mode].resize(std::size_t{size} * size * 3);
        if (Result rendered = check(tracer.render(GaussianShading::Emissive, {.samples = 1024, .seed = 3, .spectral = mode == 1}, images[mode])); !rendered)
            return rendered;
    }
    for (std::size_t channel = 0; channel < 3; ++channel) {
        std::array<double, 2> sums{};
        for (std::size_t mode = 0; mode < images.size(); ++mode)
            for (std::size_t index = channel; index < images[mode].size(); index += 3)
                sums[mode] += images[mode][index];
        const double relative = sums[1] / sums[0] - 1.0;
        std::println("spectral mean of channel {}: relative error {:+.1e}", channel, relative);
        if (!(std::abs(relative) < 5e-3))
            return std::unexpected("the spectral image differs from the RGB image");
    }
    return {};
}

}

Result testGaussianTracing(Device& device) {
    std::string failures;
    for (const Result& result : {testTable(), testForward(device), testSpectralMean(device), testGradients(device)})
        if (!result)
            failures += result.error() + "; ";
    if (!failures.empty())
        return std::unexpected(failures);
    return {};
}

}
