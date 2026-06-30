#include "Samplers/RandomSampler.h"

#include <array>
#include <cmath>
#include <cstdint>

int main()
{
    RandomState state = 0x123456789abcdef1ull;
    constexpr std::array<uint32_t, 6> expected{
        1369610156u, 2360205090u, 1914006763u,
        4290528257u, 4267313707u, 292903945u};
    for (const uint32_t value : expected)
        if (randomUint(state) != value)
            return 1;

    RandomState a = seedRandom(42u);
    RandomState b = seedRandom(42u);
    RandomState c = seedRandom(43u);
    double sum = 0.0;
    bool differs = false;
    constexpr int sampleCount = 100000;
    for (int i = 0; i < sampleCount; ++i)
    {
        const float x = randomFloat(a);
        const float y = randomFloat(b);
        const float z = randomFloat(c);
        if (x != y || x < 0.0f || x >= 1.0f)
            return 2;
        differs |= x != z;
        sum += x;
    }
    if (!differs || std::abs(sum / sampleCount - 0.5) > 0.01)
        return 3;
    return 0;
}
