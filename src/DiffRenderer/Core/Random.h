#pragma once

#include <cstdint>

namespace DiffRenderer::Core {

//! A 64-bit linear congruential generator, so a seed gives the same scene on every machine.
class Random {
public:
    explicit Random(std::uint64_t seed) : state(seed) {}

    float uniform(float low, float high);
    float normal();

private:
    std::uint64_t state;
};

}
