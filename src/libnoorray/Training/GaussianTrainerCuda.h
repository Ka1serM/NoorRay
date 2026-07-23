#include "Training/GaussianTrainer.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime_api.h>

#include "Training/GaussianTrainRenderer.h"
#include "Mesh/Assets/GaussianAsset.h"
#include "Scene/GaussianInstance.h"
#include "Scene/Scene.h"

namespace
{
void checkCuda(const cudaError_t result, const char* expression, const char* file, const int line)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": "
            + expression + ": " + cudaGetErrorString(result));
}
}

#define NR_TRAIN_CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

namespace noorray
{
namespace
{
struct DeviceBuffer
{
    float* data{};
    size_t count{};
    DeviceBuffer() = default;
    explicit DeviceBuffer(const size_t n) : count(n) { NR_TRAIN_CUDA_CHECK(cudaMalloc(&data, n * sizeof(float))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(DeviceBuffer&& other) noexcept : data(std::exchange(other.data, nullptr)), count(other.count) {}
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;
    DeviceBuffer(const DeviceBuffer&) = delete;
};

struct View
{
    std::vector<float> target;
    uint32_t width{};
    uint32_t height{};
    GaussianTrainingCamera camera;
    std::string name;
};

__global__ void l1LossGradient(const float* image, const float* target, float* gradient,
    const size_t valueCount, float* loss)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= valueCount) return;
    const float difference = image[i] - target[i];
    gradient[i] = (difference > 0.0f ? 1.0f : difference < 0.0f ? -1.0f : 0.0f)
        / static_cast<float>(valueCount);
    atomicAdd(loss, fabsf(difference) / static_cast<float>(valueCount));
}

__global__ void adam(float* parameter, const float* gradient, float* firstMoment, float* secondMoment,
    const size_t valueCount, const float learningRate, const float beta1Power, const float beta2Power)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= valueCount) return;
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    const float g = gradient[i];
    const float m = firstMoment[i] = beta1 * firstMoment[i] + (1.0f - beta1) * g;
    const float v = secondMoment[i] = beta2 * secondMoment[i] + (1.0f - beta2) * g * g;
    parameter[i] -= learningRate * (m / (1.0f - beta1Power))
        / (sqrtf(v / (1.0f - beta2Power)) + 1.0e-8f);
}

__global__ void projectGaussianShape(float* logScale, float* rotation, const size_t gaussianCount)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= gaussianCount) return;

    float* s = logScale + i * 3;
    const float center = (s[0] + s[1] + s[2]) / 3.0f;
    constexpr float maxLogDeviation = 0.69314718056f; // at most 4:1 between axes
    s[0] = fminf(fmaxf(s[0], center - maxLogDeviation), center + maxLogDeviation);
    s[1] = fminf(fmaxf(s[1], center - maxLogDeviation), center + maxLogDeviation);
    s[2] = fminf(fmaxf(s[2], center - maxLogDeviation), center + maxLogDeviation);

    float* q = rotation + i * 4;
    const float inverseNorm = rsqrtf(fmaxf(
        q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], 1.0e-12f));
    q[0] *= inverseNorm;
    q[1] *= inverseNorm;
    q[2] *= inverseNorm;
    q[3] *= inverseNorm;
}

void upload(DeviceBuffer& destination, const float* source)
{
    NR_TRAIN_CUDA_CHECK(cudaMemcpy(destination.data, source, destination.count * sizeof(float), cudaMemcpyHostToDevice));
}

struct HostParameters
{
    std::vector<float> position, logScale, rotation, opacity, color;
};

HostParameters parametersFromScene(const Scene& scene)
{
    const auto instances = scene.getGaussianInstances();
    if (instances.size() != 1 || scene.getGaussianAssets().size() != 1)
        throw std::runtime_error("Gaussian training requires exactly one Gaussian instance and asset");
    const GaussianAsset& asset = scene.getGaussianAsset(
        instances.front()->getGaussianAssetIndex());
    const size_t gaussianCount = asset.getGaussianCount();
    HostParameters result{std::vector<float>(gaussianCount * 3), std::vector<float>(gaussianCount * 3),
        std::vector<float>(gaussianCount * 4), std::vector<float>(gaussianCount),
        std::vector<float>(gaussianCount * 3)};
    for (size_t i = 0; i < gaussianCount; ++i)
    {
        const Gaussian g = asset.getGaussian(static_cast<uint32_t>(i));
        const glm::vec3 columns[3] = {g.transform[0], g.transform[1], g.transform[2]};
        glm::mat3 rotationMatrix(1.0f);
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const float scale = glm::length(columns[axis]);
            result.logScale[i * 3 + axis] = std::log(std::max(scale, 1.0e-8f));
            rotationMatrix[axis] = scale > 0.0f ? columns[axis] / scale : columns[axis];
        }
        const glm::vec3 position = g.transform[3];
        const glm::quat rotation = glm::normalize(glm::quat_cast(rotationMatrix));
        result.position[i * 3 + 0] = position.x; result.position[i * 3 + 1] = position.y;
        result.position[i * 3 + 2] = position.z;
        result.rotation[i * 4 + 0] = rotation.x; result.rotation[i * 4 + 1] = rotation.y;
        result.rotation[i * 4 + 2] = rotation.z; result.rotation[i * 4 + 3] = rotation.w;
        const float opacity = std::clamp(g.opacity, 1.0e-6f, 1.0f - 1.0e-6f);
        result.opacity[i] = std::log(opacity / (1.0f - opacity));
        const glm::vec3 rgb = g.sphericalHarmonics.count == 0
            ? glm::vec3(0.5f)
            : glm::max(glm::vec3(0.5f)
                + SphericalHarmonicsC0 * g.getShCoefficient(0), glm::vec3(0.0f));
        result.color[i * 3 + 0] = rgb.x; result.color[i * 3 + 1] = rgb.y;
        result.color[i * 3 + 2] = rgb.z;
    }
    return result;
}
}

struct GaussianTrainer::Impl
{
    DeviceBuffer position, logScale, rotation, opacity, color;
    DeviceBuffer dPosition, dLogScale, dRotation, dOpacity, dColor;
    DeviceBuffer mPosition, mLogScale, mRotation, mOpacity, mColor;
    DeviceBuffer vPosition, vLogScale, vRotation, vOpacity, vColor;
    std::unique_ptr<DeviceBuffer> target;
    std::unique_ptr<DeviceBuffer> image;
    std::unique_ptr<DeviceBuffer> imageGradient;
    std::unique_ptr<DeviceBuffer> loss;
    size_t viewBufferCapacity{};
    std::vector<View> views;
    std::mt19937_64 random;

    explicit Impl(const uint32_t n, const uint64_t seed)
        : position(n * 3), logScale(n * 3), rotation(n * 4), opacity(n), color(n * 3),
          dPosition(n * 3), dLogScale(n * 3), dRotation(n * 4), dOpacity(n), dColor(n * 3),
          mPosition(n * 3), mLogScale(n * 3), mRotation(n * 4), mOpacity(n), mColor(n * 3),
          vPosition(n * 3), vLogScale(n * 3), vRotation(n * 4), vOpacity(n), vColor(n * 3), random(seed)
    {
        for (DeviceBuffer* buffer : {&mPosition, &mLogScale, &mRotation, &mOpacity, &mColor,
             &vPosition, &vLogScale, &vRotation, &vOpacity, &vColor})
            NR_TRAIN_CUDA_CHECK(cudaMemset(buffer->data, 0, buffer->count * sizeof(float)));
    }
};

GaussianTrainer::GaussianTrainer(GaussianTrainRenderer& renderer, const uint32_t gaussianCount,
    const float* position, const float* logScale, const float* rotation,
    const float* opacityLogit, const float* colorRgb, const GaussianTrainingConfig config)
    : impl(std::make_unique<Impl>(gaussianCount, config.seed)), renderer(renderer), config(config), count(gaussianCount)
{
    upload(impl->position, position); upload(impl->logScale, logScale); upload(impl->rotation, rotation);
    upload(impl->opacity, opacityLogit); upload(impl->color, colorRgb);
}

GaussianTrainer::~GaussianTrainer() = default;

GaussianTrainer::GaussianTrainer(GaussianTrainRenderer& renderer, const Scene& scene,
    const GaussianTrainingConfig config)
    : impl(std::make_unique<Impl>(static_cast<uint32_t>(scene.getGaussianCount()), config.seed)),
      renderer(renderer), config(config), count(static_cast<uint32_t>(scene.getGaussianCount()))
{
    const HostParameters parameters = parametersFromScene(scene);
    upload(impl->position, parameters.position.data()); upload(impl->logScale, parameters.logScale.data());
    upload(impl->rotation, parameters.rotation.data()); upload(impl->opacity, parameters.opacity.data());
    upload(impl->color, parameters.color.data());
}

size_t GaussianTrainer::viewCount() const { return impl->views.size(); }

void GaussianTrainer::addView(const float* targetRgb, const uint32_t width, const uint32_t height,
    const GaussianTrainingCamera& camera, std::string name)
{
    if (!targetRgb || width == 0 || height == 0) throw std::invalid_argument("Training view is empty");
    const size_t valueCount = static_cast<size_t>(width) * height * 3;
    View view{std::vector<float>(targetRgb, targetRgb + valueCount), width, height,
        camera, std::move(name)};
    impl->views.push_back(std::move(view));
}

GaussianTrainingStep GaussianTrainer::trainStep()
{
    if (impl->views.empty()) throw std::runtime_error("No training views have been added");
    ++currentIteration;
    std::uniform_int_distribution<size_t> choose(0, impl->views.size() - 1);
    const uint32_t viewIndex = static_cast<uint32_t>(choose(impl->random));
    View& view = impl->views[viewIndex];
    const size_t pixels = static_cast<size_t>(view.width) * view.height * 3;
    if (pixels > impl->viewBufferCapacity)
    {
        // Release the previous scratch set before allocating the larger one so
        // resizing does not temporarily require both sets on the GPU.
        impl->target.reset();
        impl->image.reset();
        impl->imageGradient.reset();
        impl->loss.reset();
        impl->target = std::make_unique<DeviceBuffer>(pixels);
        impl->image = std::make_unique<DeviceBuffer>(pixels);
        impl->imageGradient = std::make_unique<DeviceBuffer>(pixels);
        impl->loss = std::make_unique<DeviceBuffer>(1);
        impl->viewBufferCapacity = pixels;
    }
    upload(*impl->target, view.target.data());

    const uint64_t seed = config.seed + currentIteration * 7919;
    renderer.renderForward(impl->position.data, impl->logScale.data, impl->rotation.data,
        impl->opacity.data, impl->color.data, count, view.camera,
        view.width, view.height, config.samplesPerPixel, seed, impl->image->data);

    NR_TRAIN_CUDA_CHECK(cudaMemset(impl->loss->data, 0, sizeof(float)));
    l1LossGradient<<<(pixels + 255) / 256, 256>>>(impl->image->data, impl->target->data,
        impl->imageGradient->data, pixels, impl->loss->data);
    NR_TRAIN_CUDA_CHECK(cudaGetLastError());
    NR_TRAIN_CUDA_CHECK(cudaDeviceSynchronize());
    renderer.renderBackward(impl->position.data, impl->logScale.data, impl->rotation.data,
        impl->opacity.data, impl->color.data, count, view.camera,
        view.width, view.height, config.samplesPerPixel, seed, impl->imageGradient->data,
        impl->dPosition.data, impl->dLogScale.data, impl->dRotation.data, impl->dOpacity.data, impl->dColor.data);

    const float beta1Power = powf(0.9f, static_cast<float>(currentIteration));
    const float beta2Power = powf(0.999f, static_cast<float>(currentIteration));
    auto update = [&](DeviceBuffer& p, DeviceBuffer& g, DeviceBuffer& m, DeviceBuffer& v, const float lr) {
        adam<<<(p.count + 255) / 256, 256>>>(p.data, g.data, m.data, v.data, p.count, lr, beta1Power, beta2Power);
    };
    update(impl->position, impl->dPosition, impl->mPosition, impl->vPosition, config.learningRatePosition);
    update(impl->logScale, impl->dLogScale, impl->mLogScale, impl->vLogScale, config.learningRateScale);
    update(impl->rotation, impl->dRotation, impl->mRotation, impl->vRotation, config.learningRateRotation);
    update(impl->opacity, impl->dOpacity, impl->mOpacity, impl->vOpacity, config.learningRateOpacity);
    update(impl->color, impl->dColor, impl->mColor, impl->vColor, config.learningRateColor);
    projectGaussianShape<<<(count + 255) / 256, 256>>>(
        impl->logScale.data, impl->rotation.data, count);
    NR_TRAIN_CUDA_CHECK(cudaGetLastError());
    NR_TRAIN_CUDA_CHECK(cudaDeviceSynchronize());
    renderer.syncScene(impl->position.data, impl->logScale.data, impl->rotation.data,
        impl->opacity.data, impl->color.data, count);
    float loss{};
    NR_TRAIN_CUDA_CHECK(cudaMemcpy(&loss, impl->loss->data, sizeof(float), cudaMemcpyDeviceToHost));
    return {currentIteration, viewIndex, loss, count};
}

std::vector<GaussianTrainingStep> GaussianTrainer::train(const uint32_t iterations)
{
    std::vector<GaussianTrainingStep> result;
    result.reserve(iterations);
    for (uint32_t i = 0; i < iterations; ++i) result.push_back(trainStep());
    return result;
}

void GaussianTrainer::exportGaussians(const std::string& path) const
{
    renderer.exportGaussians(path, impl->position.data, impl->logScale.data, impl->rotation.data,
        impl->opacity.data, impl->color.data, count);
}
}
