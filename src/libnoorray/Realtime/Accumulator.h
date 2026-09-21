#pragma once

#include <noorrhi/noorrhi.hpp>

#include "Shared/RealtimeArgs.h"

// Folds each frame's output beauty into a running average, as the offline
// tracer does, so a still view converges. Frame sample index 0 restarts it.
class Accumulator
{
public:
    explicit Accumulator(noorrhi::Device& device);

    // `args` is what `root` points at.
    void record(const nr::graphics::RealtimeArgs& args, nr::graphics::RealtimeRoot root) const;

private:
    noorrhi::ComputePipeline pipeline_;
};
