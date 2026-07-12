#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>

class GaussianTrainRenderer;
class Scene;

namespace noorray
{

struct GaussianTrainingConfig
{
    uint32_t samplesPerPixel{1};
    float learningRatePosition{1.6e-4f};
    float learningRateScale{5.0e-3f};
    float learningRateRotation{1.0e-3f};
    float learningRateOpacity{5.0e-2f};
    float learningRateColor{2.5e-3f};
    uint64_t seed{0};
};

struct GaussianTrainingStep
{
    uint64_t iteration{};
    uint32_t viewIndex{};
    float loss{};
    uint32_t gaussianCount{};
};

struct GaussianTrainingCamera
{
    glm::mat4 cameraToWorld{1.0f};
    float fx{}, fy{}, cx{}, cy{};
};

class GaussianTrainer
{
public:
    GaussianTrainer(GaussianTrainRenderer& renderer, uint32_t gaussianCount,
        const float* position, const float* logScale, const float* rotation,
        const float* opacityLogit, const float* colorRgb, GaussianTrainingConfig config = {});
    GaussianTrainer(GaussianTrainRenderer& renderer, const Scene& scene,
        GaussianTrainingConfig config = {});
    ~GaussianTrainer();

    GaussianTrainer(const GaussianTrainer&) = delete;
    GaussianTrainer& operator=(const GaussianTrainer&) = delete;

    void addView(const float* targetRgb, uint32_t width, uint32_t height,
        const GaussianTrainingCamera& camera, std::string name = {});
    GaussianTrainingStep trainStep();
    std::vector<GaussianTrainingStep> train(uint32_t iterations);
    void exportGaussians(const std::string& path) const;

    uint32_t gaussianCount() const { return count; }
    uint64_t iteration() const { return currentIteration; }
    size_t viewCount() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    GaussianTrainRenderer& renderer;
    GaussianTrainingConfig config;
    uint32_t count{};
    uint64_t currentIteration{};
};

}
