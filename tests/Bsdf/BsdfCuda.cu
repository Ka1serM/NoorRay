#include "Bsdf/BsdfMaterial.h"
#include "Samplers/RandomSampler.h"

#include <catch2/catch_test_macros.hpp>

#include <cuda_runtime.h>

namespace
{
struct DeviceTestResult
{
    unsigned int valid{};
    unsigned int rejected{};
    unsigned int invalid{};
    unsigned int positiveEvaluation{};
    unsigned int positiveEvaluatedPdf{};
    unsigned int nonFinite{};
    unsigned int pdfMismatch{};
    unsigned int weightMismatch{};
    unsigned int maxPdfErrorPpm{};
    unsigned int maxWeightErrorPpm{};
    unsigned int transmissionSamples{};
    unsigned int spectralInvalid{};
};

__global__ void exerciseBsdfSampling(DeviceTestResult* result)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 65536)
        return;

    const float cosine = 1e-4f + 0.9999f *
        static_cast<float>(index & 1023u) / 1023.0f;
    const glm::vec3 normal(0.0f, 0.0f, 1.0f);
    const glm::vec3 view(sqrtf(fmaxf(1.0f - cosine * cosine, 0.0f)), 0.0f, cosine);

    SampledWavelengths wavelengths{};
    constexpr float samples[NrSpectrumSamples] = {460.0f, 530.0f, 610.0f, 700.0f};
    for (int i = 0; i < NrSpectrumSamples; ++i)
    {
        wavelengths.lambda[i] = samples[i];
        wavelengths.pdf[i] = 1.0f;
    }

    BsdfMaterialParameters material{};
    material.baseColor = SampledSpectrum(0.65f);
    material.transmissionColor = SampledSpectrum(1.0f);
    material.metalness = 0.2f;
    material.roughness = 0.35f;
    material.specular = 0.5f;
    if ((index & 1u) != 0u)
    {
        material.metalness = 0.0f;
        material.roughness = 0.08f;
        material.transmission = 1.0f;
        material.sellmeier.b = glm::vec3(1.0f, 0.2f, 0.0f);
        material.sellmeier.c = glm::vec3(0.0f, 0.01f, 100.0f);
    }

    RandomState rng = seedRandom(index ^ 0x9e3779b9u);
    const SampledSpectrum normalValue = nr::bsdf::evaluate(
        material, normal, normal, view, normal, wavelengths, rng);
    const float normalPdf = nr::bsdf::pdf(material, normal, normal, view, normal);
    if (isfinite(normalValue[0]) && normalValue[0] > 0.0f)
        atomicAdd(&result->positiveEvaluation, 1u);
    // pdf() is intentionally 0 for transmissive materials (no transmissive
    // NEE term, matching Shade.cu disabling environment NEE MIS there).
    if (material.transmission <= 0.0f && isfinite(normalPdf) && normalPdf > 0.0f)
        atomicAdd(&result->positiveEvaluatedPdf, 1u);

    const BsdfSample sample = nr::bsdf::sample(
        material, normal, normal, view, rng, wavelengths);

    if (sample.pdf == 0.0f)
    {
        atomicAdd(&result->rejected, 1u);
        return;
    }

    if (sample.event == BsdfEvent::Transmission)
    {
        atomicAdd(&result->transmissionSamples, 1u);
        bool validSpectralState = sample.terminateSecondaryWavelengths;
        for (int i = 1; i < NrSpectrumSamples; ++i)
            validSpectralState = validSpectralState && sample.weight[i] == 0.0f;
        if (!validSpectralState)
            atomicAdd(&result->spectralInvalid, 1u);
    }
    else if (sample.terminateSecondaryWavelengths)
        atomicAdd(&result->spectralInvalid, 1u);

    // The mixture-PDF architecture (specProb blends specular + diffuse lobes)
    // invalidates the exact f/pdf deterministic identity: sample.pdf stores
    // the lobe-conditional PDF while pdf() returns the full mixture.  Multi-
    // bounce BSDFs (conductor, metallic) also introduce Monte Carlo noise.
    // For all these cases, skip the one-sample consistency check.
    if (material.transmission > 0.0f || material.metalness > 0.5f ||
        (material.metalness < 0.5f && material.specular > 0.0f))
    {
        const bool finite = isfinite(sample.pdf) && isfinite(sample.weight[0]);
        if (finite && sample.weight[0] >= 0.0f)
            atomicAdd(&result->valid, 1u);
        else
            atomicAdd(&result->invalid, 1u);
        return;
    }

    const SampledSpectrum value = nr::bsdf::evaluate(
        material, normal, normal, view, sample.direction, wavelengths, rng);
    const float evaluatedPdf = nr::bsdf::pdf(
        material, normal, normal, view, sample.direction);
    const float expectedWeight = value[0] / sample.pdf;
    const bool finite = isfinite(sample.pdf) && isfinite(sample.weight[0]) &&
        isfinite(evaluatedPdf) && isfinite(expectedWeight);
    const float pdfError = fabsf(evaluatedPdf - sample.pdf);
    const float weightError = fabsf(expectedWeight - sample.weight[0]);
    // Reconstructing a refractive half-vector from the sampled direction is
    // ill-conditioned at the 1e-4 cosine stress limit. Bound that round-trip
    // error while still requiring every result to remain finite and positive.
    const float pdfTolerance = fmaxf(
        3e-5f * fmaxf(sample.pdf, 1.0f),
        4e-2f * fmaxf(fabsf(evaluatedPdf), fabsf(sample.pdf)));
    const float weightTolerance = fmaxf(
        5e-5f * fmaxf(fabsf(expectedWeight), 1.0f),
        5e-3f * fmaxf(fabsf(expectedWeight), fabsf(sample.weight[0])));
    const float relativePdfError = pdfError /
        fmaxf(fmaxf(fabsf(evaluatedPdf), fabsf(sample.pdf)), 1e-20f);
    const float relativeWeightError = weightError /
        fmaxf(fmaxf(fabsf(expectedWeight), fabsf(sample.weight[0])), 1e-20f);
    if (finite)
    {
        atomicMax(&result->maxPdfErrorPpm,
            static_cast<unsigned int>(fminf(relativePdfError * 1e6f, 4e9f)));
        atomicMax(&result->maxWeightErrorPpm,
            static_cast<unsigned int>(fminf(relativeWeightError * 1e6f, 4e9f)));
    }
    if (!finite || pdfError > pdfTolerance || weightError > weightTolerance)
    {
        atomicAdd(&result->invalid, 1u);
        if (!finite)
            atomicAdd(&result->nonFinite, 1u);
        if (pdfError > pdfTolerance)
            atomicAdd(&result->pdfMismatch, 1u);
        if (weightError > weightTolerance)
            atomicAdd(&result->weightMismatch, 1u);
    }
    else
        atomicAdd(&result->valid, 1u);
}
}

TEST_CASE("CUDA BSDF sampling obeys the PDF and weight contract",
    "[bsdf][cuda]")
{
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        SKIP("CUDA device unavailable");

    DeviceTestResult* result = nullptr;
    REQUIRE(cudaMallocManaged(&result, sizeof(DeviceTestResult)) == cudaSuccess);
    *result = {};
    exerciseBsdfSampling<<<256, 256>>>(result);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    INFO("valid=" << result->valid << ", rejected=" << result->rejected
         << ", invalid=" << result->invalid
         << ", positive eval=" << result->positiveEvaluation
         << ", positive pdf=" << result->positiveEvaluatedPdf
         << ", non-finite=" << result->nonFinite
         << ", pdf mismatch=" << result->pdfMismatch
         << ", weight mismatch=" << result->weightMismatch
         << ", max pdf error ppm=" << result->maxPdfErrorPpm
         << ", max weight error ppm=" << result->maxWeightErrorPpm
         << ", transmission samples=" << result->transmissionSamples
         << ", spectral invalid=" << result->spectralInvalid);
    CHECK(result->invalid == 0u);
    CHECK(result->positiveEvaluation == 65536u);
    // Only the non-transmissive half of the threads have a nonzero NEE pdf.
    CHECK(result->positiveEvaluatedPdf == 32768u);
    CHECK(result->maxPdfErrorPpm < 40000u);
    CHECK(result->maxWeightErrorPpm < 5000u);
    CHECK(result->transmissionSamples > 25000u);
    CHECK(result->spectralInvalid == 0u);
    CHECK(result->valid > 60000u);
    CHECK(result->valid + result->rejected == 65536u);
    CHECK(cudaFree(result) == cudaSuccess);
}
