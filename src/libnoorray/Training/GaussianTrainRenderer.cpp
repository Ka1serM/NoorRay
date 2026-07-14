#include "Training/GaussianTrainRenderer.h"
#include "Training/GaussianTrainer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>

#include <gf/core/gauss_ir.h>
#include <gf/io/ply.h>
#include <gf/io/sog.h>

#include "CUDA/Checks.h"
#include "Mesh/GaussianAsset.h"
#include "Training/GaussianTrainData.h"
#include "Raytracing/Raytracer.h"
#include "Scene/CoordinateSystem.h"
#include "Scene/GaussianInstance.h"
#include "Scene/Scene.h"

GaussianTrainRenderer::GaussianTrainRenderer(Raytracer& raytracer, Scene& scene)
    : raytracer(raytracer), scene(scene), stream(raytracer.getCudaStream())
{
}

GaussianTrainRenderer::~GaussianTrainRenderer()
{
    if (stream != nullptr)
        cudaStreamSynchronize(stream);
}

void GaussianTrainRenderer::syncScene(
    const float* positionDevice, const float* logScaleDevice, const float* rotationDevice,
    const float* opacityLogitDevice, const float* colorRgbDevice, const uint32_t gaussianCount)
{
    const auto& instances = scene.getGaussianInstances();
    if (instances.size() != 1 || scene.getGaussianAssets().size() != 1)
        throw std::runtime_error("Gaussian training requires exactly one Gaussian instance and asset");
    GaussianAsset& asset = scene.getGaussianAsset(instances.front()->getGaussianAssetIndex());
    asset.resizeGaussians(gaussianCount);

    std::vector<glm::vec3> positions(gaussianCount), logScales(gaussianCount), colors(gaussianCount);
    std::vector<glm::vec4> rotations(gaussianCount);
    std::vector<float> opacityLogits(gaussianCount);
    NR_GPU_CHECK(cudaMemcpy(positions.data(), positionDevice, positions.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(logScales.data(), logScaleDevice, logScales.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(rotations.data(), rotationDevice, rotations.size() * sizeof(glm::vec4), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(opacityLogits.data(), opacityLogitDevice, opacityLogits.size() * sizeof(float), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(colors.data(), colorRgbDevice, colors.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < gaussianCount; ++i)
    {
        Gaussian& gaussian = asset.getGaussians()[i];
        const glm::vec4 r = rotations[i];
        const glm::mat3 rotation = glm::mat3_cast(glm::normalize(glm::quat(r.w, r.x, r.y, r.z)));
        const glm::vec3 scale = glm::exp(logScales[i]);
        gaussian.transform = glm::mat4x3(rotation[0] * scale.x, rotation[1] * scale.y,
            rotation[2] * scale.z, positions[i]);
        gaussian.opacity = 1.0f / (1.0f + std::exp(-opacityLogits[i]));
        if (gaussian.shCoeffCount == 0) gaussian.shCoeffCount = 1;
        constexpr float C0 = 0.28209479177387814f;
        gaussian.shCoeffs[0] = (colors[i] - glm::vec3(0.5f)) / C0;
    }
    raytracer.updateTLAS();
}

void GaussianTrainRenderer::bakeTransformsAndUpdateTlas(
    const float* positionDevice, const float* logScaleDevice, const float* rotationDevice,
    const uint32_t gaussianCount)
{
    const auto& instances = scene.getGaussianInstances();
    if (instances.size() != 1 || scene.getGaussianAssets().size() != 1)
    {
        throw std::runtime_error(
            "GaussianTrainRenderer v1 only supports a scene with exactly one "
            "Gaussian instance/asset");
    }
    GaussianAsset& asset = scene.getGaussianAsset(instances.front()->getGaussianAssetIndex());
    if (asset.getGaussianCount() != gaussianCount)
    {
        // Densification/pruning changes the trainable parameter count
        // between iterations; grow/shrink the backing storage to match
        // (see GaussianAsset::resizeGaussians). Only `.transform` is
        // populated for the new entries below -- opacity/SH coefficients
        // stay zeroed, which is fine since the training kernels never read
        // them (they use opacityLogit/colorRgb passed in directly).
        asset.resizeGaussians(gaussianCount);
    }

    std::vector<glm::vec3> position(gaussianCount);
    std::vector<glm::vec3> logScale(gaussianCount);
    std::vector<glm::vec4> rotation(gaussianCount);
    NR_GPU_CHECK(cudaMemcpy(position.data(), positionDevice,
        position.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(logScale.data(), logScaleDevice,
        logScale.size() * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(rotation.data(), rotationDevice,
        rotation.size() * sizeof(glm::vec4), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < gaussianCount; ++i)
    {
        const glm::quat q = glm::normalize(glm::quat(
            rotation[i].w, rotation[i].x, rotation[i].y, rotation[i].z));
        const glm::mat3 R = glm::mat3_cast(q);
        const glm::vec3 scale = glm::exp(logScale[i]);
        asset.getGaussians()[i].transform = glm::mat4x3(
            R[0] * scale.x, R[1] * scale.y, R[2] * scale.z, position[i]);
    }
    NR_GPU_CHECK(cudaStreamSynchronize(stream));

    raytracer.updateTLAS();
}

void GaussianTrainRenderer::renderForward(
    const float* positionDevice, const float* logScaleDevice,
    const float* rotationDevice, const float* opacityLogitDevice,
    const float* colorRgbDevice, const uint32_t gaussianCount,
    const noorray::GaussianTrainingCamera& camera,
    const uint32_t width, const uint32_t height, const uint32_t samplesPerPixel, const uint64_t seed,
    float* outputColorDevice)
{
    bakeTransformsAndUpdateTlas(positionDevice, logScaleDevice, rotationDevice, gaussianCount);

    GaussianTrainParams trainParams{};
    trainParams.tlas = raytracer.getGaussianTlasHandle();
    trainParams.position = reinterpret_cast<const glm::vec3*>(positionDevice);
    trainParams.logScale = reinterpret_cast<const glm::vec3*>(logScaleDevice);
    trainParams.rotation = reinterpret_cast<const glm::vec4*>(rotationDevice);
    trainParams.opacityLogit = opacityLogitDevice;
    trainParams.colorRgb = reinterpret_cast<const glm::vec3*>(colorRgbDevice);
    trainParams.gaussianCount = gaussianCount;
    trainParams.cameraToWorld = camera.cameraToWorld;
    trainParams.fx = camera.fx; trainParams.fy = camera.fy;
    trainParams.cx = camera.cx; trainParams.cy = camera.cy;
    trainParams.width = width;
    trainParams.height = height;
    trainParams.samplesPerPixel = samplesPerPixel;
    trainParams.seed = seed;
    trainParams.outputColor = reinterpret_cast<glm::vec3*>(outputColorDevice);

    raytracer.renderGaussianTrainForward(trainParams, width, height);
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void GaussianTrainRenderer::renderBackward(
    const float* positionDevice, const float* logScaleDevice,
    const float* rotationDevice, const float* opacityLogitDevice,
    const float* colorRgbDevice, const uint32_t gaussianCount,
    const noorray::GaussianTrainingCamera& camera,
    const uint32_t width, const uint32_t height, const uint32_t samplesPerPixel, const uint64_t seed,
    const float* dLdImageDevice,
    float* dPositionDevice, float* dLogScaleDevice, float* dRotationDevice,
    float* dOpacityLogitDevice, float* dColorRgbDevice)
{
    // Reuses the TLAS built by the preceding renderForward() call -- the
    // Gaussian parameters must not have changed in between.
    NR_GPU_CHECK(cudaMemsetAsync(dPositionDevice, 0, sizeof(float) * 3 * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(dLogScaleDevice, 0, sizeof(float) * 3 * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(dRotationDevice, 0, sizeof(float) * 4 * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(dOpacityLogitDevice, 0, sizeof(float) * gaussianCount, stream));
    NR_GPU_CHECK(cudaMemsetAsync(dColorRgbDevice, 0, sizeof(float) * 3 * gaussianCount, stream));

    GaussianTrainParams trainParams{};
    trainParams.tlas = raytracer.getGaussianTlasHandle();
    trainParams.position = reinterpret_cast<const glm::vec3*>(positionDevice);
    trainParams.logScale = reinterpret_cast<const glm::vec3*>(logScaleDevice);
    trainParams.rotation = reinterpret_cast<const glm::vec4*>(rotationDevice);
    trainParams.opacityLogit = opacityLogitDevice;
    trainParams.colorRgb = reinterpret_cast<const glm::vec3*>(colorRgbDevice);
    trainParams.gaussianCount = gaussianCount;
    trainParams.cameraToWorld = camera.cameraToWorld;
    trainParams.fx = camera.fx; trainParams.fy = camera.fy;
    trainParams.cx = camera.cx; trainParams.cy = camera.cy;
    trainParams.width = width;
    trainParams.height = height;
    trainParams.samplesPerPixel = samplesPerPixel;
    trainParams.seed = seed;
    trainParams.dLdImage = reinterpret_cast<const glm::vec3*>(dLdImageDevice);
    trainParams.dPosition = reinterpret_cast<glm::vec3*>(dPositionDevice);
    trainParams.dLogScale = reinterpret_cast<glm::vec3*>(dLogScaleDevice);
    trainParams.dRotation = reinterpret_cast<glm::vec4*>(dRotationDevice);
    trainParams.dOpacityLogit = dOpacityLogitDevice;
    trainParams.dColorRgb = reinterpret_cast<glm::vec3*>(dColorRgbDevice);

    raytracer.renderGaussianTrainBackward(trainParams, width, height);
    NR_GPU_CHECK(cudaStreamSynchronize(stream));
}

void GaussianTrainRenderer::exportGaussians(
    const std::string& path,
    const float* positionDevice, const float* logScaleDevice, const float* rotationDevice,
    const float* opacityLogitDevice, const float* colorRgbDevice, const uint32_t gaussianCount)
{
    std::vector<glm::vec3> position(gaussianCount);
    std::vector<glm::vec3> logScale(gaussianCount);
    std::vector<glm::vec4> rotation(gaussianCount);
    std::vector<float> opacityLogit(gaussianCount);
    std::vector<glm::vec3> colorRgb(gaussianCount);
    NR_GPU_CHECK(cudaMemcpy(position.data(), positionDevice, gaussianCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(logScale.data(), logScaleDevice, gaussianCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(rotation.data(), rotationDevice, gaussianCount * sizeof(glm::vec4), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(opacityLogit.data(), opacityLogitDevice, gaussianCount * sizeof(float), cudaMemcpyDeviceToHost));
    NR_GPU_CHECK(cudaMemcpy(colorRgb.data(), colorRgbDevice, gaussianCount * sizeof(glm::vec3), cudaMemcpyDeviceToHost));

    // Inverse of GaussianAsset.cpp's import conversion: YDownZForwardSpace's
    // basis flip is its own inverse (diag(1,-1,-1)), so the same call
    // converts OpenGL-space back to the raw-3DGS convention.
    static constexpr float SH_C0 = 0.28209479177387814f;
    gf::GaussianCloudIR ir;
    ir.numPoints = static_cast<int32_t>(gaussianCount);
    ir.positions.resize(gaussianCount * 3);
    ir.scales.resize(gaussianCount * 3);
    ir.rotations.resize(gaussianCount * 4);
    ir.alphas.resize(gaussianCount);
    ir.colors.resize(gaussianCount * 3);

    for (uint32_t i = 0; i < gaussianCount; ++i)
    {
        const glm::vec3 posSource = nr::coords::toOpenGlVector(position[i], nr::coords::YDownZForwardSpace);
        ir.positions[i * 3 + 0] = posSource.x;
        ir.positions[i * 3 + 1] = posSource.y;
        ir.positions[i * 3 + 2] = posSource.z;

        ir.scales[i * 3 + 0] = logScale[i].x;
        ir.scales[i * 3 + 1] = logScale[i].y;
        ir.scales[i * 3 + 2] = logScale[i].z;

        const glm::vec4 rot = rotation[i];
        const glm::mat3 Rgl = glm::mat3_cast(glm::normalize(glm::quat(rot.w, rot.x, rot.y, rot.z)));
        const glm::mat3 Rsrc(
            nr::coords::toOpenGlVector(Rgl[0], nr::coords::YDownZForwardSpace),
            nr::coords::toOpenGlVector(Rgl[1], nr::coords::YDownZForwardSpace),
            nr::coords::toOpenGlVector(Rgl[2], nr::coords::YDownZForwardSpace));
        const glm::quat qSrc = glm::normalize(glm::quat_cast(Rsrc));
        ir.rotations[i * 4 + 0] = qSrc.w;
        ir.rotations[i * 4 + 1] = qSrc.x;
        ir.rotations[i * 4 + 2] = qSrc.y;
        ir.rotations[i * 4 + 3] = qSrc.z;

        ir.alphas[i] = opacityLogit[i];

        ir.colors[i * 3 + 0] = (colorRgb[i].x - 0.5f) / SH_C0;
        ir.colors[i * 3 + 1] = (colorRgb[i].y - 0.5f) / SH_C0;
        ir.colors[i * 3 + 2] = (colorRgb[i].z - 0.5f) / SH_C0;
    }

    const std::filesystem::path filePath(path);
    const std::string extension = filePath.extension().string();
    std::unique_ptr<gf::IGaussWriter> writer;
    if (extension == ".ply")
        writer = gf::MakePlyWriter();
    else if (extension == ".sog")
        writer = gf::MakeSogWriter();
    else
        throw std::runtime_error("Unsupported Gaussian export format: " + path);

    const gf::Expected<std::vector<uint8_t>> result = writer->Write(ir, gf::WriteOptions{});
    if (!result)
        throw std::runtime_error("Failed to write Gaussian file: " + result.error().message);

    std::ofstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Failed to open output file: " + path);
    file.write(reinterpret_cast<const char*>(result.value().data()),
        static_cast<std::streamsize>(result.value().size()));
}
