#pragma once

#include <cstdint>
#include <expected>
#include <string_view>

#include "DiffRenderer/Core/Image.h"
#include "NoorRHI/NoorRHI.h"

namespace DiffRenderer::Algorithms {

struct Settings {
    std::uint64_t seed = 1;
    //! Multiplies the learning rate of every parameter group.
    float learningRateScale = 1.0f;
    //! Turns the treatment of the discontinuities on or off, for the algorithms that have one.
    bool boundaryTerm = true;
};

struct Description {
    std::string_view id;
    std::string_view name;
    std::string_view representation;
    std::string_view parameters;
    std::string_view discontinuities;
    std::string_view reference;
    bool hasBoundaryTerm = false;
};

struct Step {
    double loss = 0.0;
    //! The mean absolute distance of the optimized parameters from the true ones.
    double parameterError = 0.0;
};

//! One differentiable renderer with its test scene: it renders a target from the true parameters and optimizes perturbed parameters toward it.
class Algorithm {
public:
    virtual ~Algorithm() = default;

    virtual const Description& getDescription() const = 0;
    //! Renders the target and starts again from the perturbed parameters of the seed.
    [[nodiscard]] virtual std::expected<void, NoorRHI::Error> open(NoorRHI::Device& device, const Settings& settings) = 0;
    //! Renders the current parameters, takes the loss against the target, back-propagates it and takes one optimizer step.
    [[nodiscard]] virtual std::expected<Step, NoorRHI::Error> step() = 0;
    virtual const Core::Image& getTarget() const = 0;
    //! The image of the parameters before the last step.
    virtual const Core::Image& getCurrent() const = 0;
};

}
