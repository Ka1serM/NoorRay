#pragma once

#include <span>
#include <vector>

namespace DiffRenderer::Core {

//! Adam (Kingma and Ba 2015) for one group of parameters with one learning rate.
class Adam {
public:
    explicit Adam(float learningRate = 1e-3f) : learningRate(learningRate) {}

    void setLearningRate(float rate) { learningRate = rate; }
    void reset();
    void step(std::span<float> values, std::span<const float> gradients);

private:
    float learningRate;
    std::vector<float> first;
    std::vector<float> second;
    int iteration = 0;
};

}
