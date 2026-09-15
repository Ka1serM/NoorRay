#include "Environment.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <glm/gtc/matrix_inverse.hpp>

#include "Texture/Texture.h"

static constexpr float kPi = 3.14159265f;

Environment::Environment()
{
    // The record is value-initialised, so any default that is not zero has to
    // be established here rather than in a member initialiser.
    data.color = glm::vec3(1.0f);
    setEquirectangularMapping();
    updateDerivedSettings();
}

Environment::~Environment()
{
    destroyCdf();
}

glm::mat3 Environment::environmentFromWorld() const
{
    return glm::mat3(glm::vec3(data.environmentFromWorld[0]),
                     glm::vec3(data.environmentFromWorld[1]),
                     glm::vec3(data.environmentFromWorld[2]));
}

void Environment::releaseGpu() noexcept
{
    hdriImage = {};
    cdfImage = {};
    // Keeps `data`, which is this object's own state, not a cache of it.
    release();
}

void Environment::destroyCdf() noexcept
{
    data.cdfTexture = 0;
    data.cdfWidth = 0;
    data.cdfHeight = 0;
    cdfDirty = 1;
}

void Environment::updateDerivedSettings()
{
    const float rotationRadians = rotation * (kPi / 180.f);
    data.rotationSin = std::sin(rotationRadians);
    data.rotationCos = std::cos(rotationRadians);
    data.lightingExposureScale = lightingExposure;
    // Intensity is the base multiplier for both illumination and the visible
    // background. Visible Exposure is an additional camera-only stop offset.
    data.visibleExposureScale = lightingExposure * std::pow(2.f, visibleExposure);
    const float luminance = std::max(0.2126f * data.color.r
        + 0.7152f * data.color.g + 0.0722f * data.color.b, 0.0f);
    data.importanceWeight =
        4.0f * kPi * luminance * std::max(data.lightingExposureScale, 0.0f);
    cdfDirty = 1;
}

void Environment::setHdriTexture(const Texture& texture)
{
    if (texture.getSceneIndex() < 0)
        throw std::invalid_argument("The HDRI texture must be added to a Scene first");
    textureIndex = texture.getSceneIndex();
    cdfDirty = 1;
}

void Environment::clearHdriTexture()
{
    textureIndex = -1;
    hdriImage = {};
    cdfImage = {};
    release();
    cdfDirty = 1;
}

void Environment::setMapping(const EnvironmentMapping newMapping, const glm::mat3& transform)
{
    data.mapping = static_cast<std::int32_t>(newMapping);
    for (int column = 0; column < 3; ++column)
        data.environmentFromWorld[column] = float4(transform[column], 0.0f);
    worldFromEnvironment = glm::inverse(transform);
    cdfDirty = 1;
}

void Environment::setEquirectangularMapping(const glm::mat3& transform)
{
    setMapping(EnvironmentMapping::Equirectangular, transform);
}

void Environment::setEqualAreaMapping(const glm::mat3& transform)
{
    setMapping(EnvironmentMapping::EqualArea, transform);
}

std::vector<float> Environment::computeCdf(
    const float* hdr, const int w, const int h, const EnvironmentMapping mapping)
{
    std::vector<float> out(w * h * 4, 0.0f);
    std::vector<float> rowIntegrals(h, 0.0f);
    // Flat row-major storage to avoid pointer chasing across threads.
    std::vector<float> rowWeights(h * w, 0.0f);

    // Phase 1: luminance × sin(θ) weights — parallel over rows.
    tbb::parallel_for(tbb::blocked_range<int>(0, h),
        [&](const tbb::blocked_range<int>& r) {
            for (int y = r.begin(); y != r.end(); ++y) {
                const float solidAngleWeight = mapping == EnvironmentMapping::EqualArea
                    ? 1.0f : std::sin((y + 0.5f) / float(h) * kPi);
                float integral = 0.0f;
                for (int x = 0; x < w; ++x) {
                    const int src = (y * w + x) * 4;
                    const float lum = 0.2126f * hdr[src]
                                    + 0.7152f * hdr[src + 1]
                                    + 0.0722f * hdr[src + 2];
                    const float weight = lum * solidAngleWeight;
                    rowWeights[y * w + x] = weight;
                    integral += weight;
                }
                rowIntegrals[y] = integral;
            }
        });

    // Phase 2: marginal CDF — sequential prefix scan across rows.
    const float total    = std::accumulate(rowIntegrals.begin(), rowIntegrals.end(), 0.0f);
    const float totalInv = total > 0.0f ? 1.0f / total : 0.0f;
    float margAcc = 0.0f;
    for (int y = 0; y < h; ++y) {
        margAcc += rowIntegrals[y] * totalInv;
        out[y * w * 4 + 1] = margAcc;   // G channel, column 0
    }

    // Phase 3: conditional CDF + PDF per row — parallel over rows.
    tbb::parallel_for(tbb::blocked_range<int>(0, h),
        [&](const tbb::blocked_range<int>& r) {
            for (int y = r.begin(); y != r.end(); ++y) {
                const float sinTheta = mapping == EnvironmentMapping::EqualArea ? 1.0f
                    : std::max(std::sin((y + 0.5f) / float(h) * kPi), 1e-6f);
                const float rowNorm  = rowIntegrals[y] > 0.0f ? 1.0f / rowIntegrals[y] : 0.0f;
                const float pdfScale = mapping == EnvironmentMapping::EqualArea
                    ? float(w) * float(h) / (4.0f * kPi)
                    : float(w) * float(h) / (2.0f * kPi * kPi * sinTheta);
                float condAcc = 0.0f;
                for (int x = 0; x < w; ++x) {
                    const float weight = rowWeights[y * w + x];
                    condAcc += weight * rowNorm;
                    const int dst = (y * w + x) * 4;
                    out[dst]     = condAcc;                         // R = conditional CDF
                    out[dst + 3] = weight * totalInv * pdfScale;   // A = PDF
                }
            }
        });

    return out;
}

void Environment::uploadImages(gpu::Device& device, const Texture* hdri)
{
    hdriImage = {};
    cdfImage = {};
    if (hdri != nullptr) {
        const std::vector<float>& pixels = hdri->getPixels();
        const int width = hdri->getWidth();
        const int height = hdri->getHeight();
        const std::size_t valueCount = static_cast<std::size_t>(width)
            * static_cast<std::size_t>(height) * 4u;
        if (width <= 0 || height <= 0 || pixels.size() < valueCount)
            throw std::runtime_error("invalid HDRI pixel storage");

        hdriImage = device.image<std::byte>(width, height,
            gpu::ImageUsage::Sampled, gpu::ImageFormat::Rgba32Float);
        hdriImage.upload(std::span<const std::byte>(std::as_bytes(std::span(pixels))));

        std::vector<float> cdf = computeCdf(pixels.data(), width, height, mapping());
        if (cdf.size() != valueCount)
            throw std::runtime_error("invalid HDRI importance CDF");
        cdfImage = device.image<std::byte>(width, height,
            gpu::ImageUsage::Sampled, gpu::ImageFormat::Rgba32Float);
        cdfImage.upload(std::span<const std::byte>(std::as_bytes(std::span(cdf))));
    }
    uploadRecord(device);
}

void Environment::uploadRecord(gpu::Device& device)
{
    if (!*this)
        allocate(device);

    // Every other field is already live in `data`; only the image handles are
    // resolved here, because they exist just after uploadImages().
    data.texture = hdriImage ? hdriImage.sampled_handle().value
        : nr::graphics::EnvironmentNoTexture;
    data.cdfTexture = cdfImage ? cdfImage.sampled_handle().value
        : nr::graphics::EnvironmentNoTexture;
    commit();
}
