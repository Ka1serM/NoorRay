#include "Environment.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "Vulkan/Texture.h"

static constexpr float kPi = 3.14159265f;

Environment::Environment()
{
    updateDerivedSettings();
}

Environment::~Environment()
{
    destroyCdf();
}

void Environment::destroyCdf() noexcept
{
    cdfTexture.reset();
    cdfWidth = 0;
    cdfHeight = 0;
    cdfDirty = 1;
}

void Environment::updateDerivedSettings()
{
    const float rotationRadians = rotation * (kPi / 180.f);
    rotationSin = std::sin(rotationRadians);
    rotationCos = std::cos(rotationRadians);
    lightingExposureScale = lightingExposure;
    visibleExposureScale = std::pow(2.f, visibleExposure);
    const float luminance = std::max(
        0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b, 0.0f);
    importanceWeight = 4.0f * kPi * luminance * std::max(lightingExposureScale, 0.0f);
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
    cdfDirty = 1;
}

std::vector<float> Environment::computeCdf(const float* hdr, const int w, const int h)
{
    std::vector<float> out(w * h * 4, 0.0f);
    std::vector<float> rowIntegrals(h, 0.0f);
    // Flat row-major storage to avoid pointer chasing across threads.
    std::vector<float> rowWeights(h * w, 0.0f);

    // Phase 1: luminance × sin(θ) weights — parallel over rows.
    tbb::parallel_for(tbb::blocked_range<int>(0, h),
        [&](const tbb::blocked_range<int>& r) {
            for (int y = r.begin(); y != r.end(); ++y) {
                const float sinTheta = std::sin((y + 0.5f) / float(h) * kPi);
                float integral = 0.0f;
                for (int x = 0; x < w; ++x) {
                    const int src = (y * w + x) * 4;
                    const float lum = 0.2126f * hdr[src]
                                    + 0.7152f * hdr[src + 1]
                                    + 0.0722f * hdr[src + 2];
                    const float weight = lum * sinTheta;
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
                const float sinTheta = std::max(std::sin((y + 0.5f) / float(h) * kPi), 1e-6f);
                const float rowNorm  = rowIntegrals[y] > 0.0f ? 1.0f / rowIntegrals[y] : 0.0f;
                const float pdfScale = float(w) * float(h) / (2.0f * kPi * kPi * sinTheta);
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
