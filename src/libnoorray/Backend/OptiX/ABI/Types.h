#pragma once

#include <cstdint>

#include "Backend/CUDA/Annotations.h"
#include "Rendering/Ray.h"
#include "Materials/Shading/Spectrum.h"
#include "Rendering/Sampling/RandomSampler.h"

#if defined(NR_GPU_CODE)
#include <optix_device.h>
#endif

static constexpr uint32_t InvalidIndex = ~0u;
// AOV Cryptomatte queries reserve this sample value to request analytic,
// full-opacity Gaussian intersections. It is only interpreted while an AOV
// refresh is active, so regular path samples remain unaffected.
static constexpr uint32_t OpaqueAovGaussianSample = InvalidIndex - 1u;
// Payload-2 marker used only by the opaque Cryptomatte query. This keeps a
// beauty shadow ray's random payload from being mistaken for that query.
static constexpr uint32_t OpaqueAovQueryMarker = InvalidIndex - 2u;
static constexpr uint8_t MeshVisibility = uint8_t{1} << 0u;
static constexpr uint8_t GaussianVisibility = uint8_t{1} << 1u;
static constexpr uint8_t AnalyticLightVisibility = uint8_t{1} << 2u;
static constexpr uint8_t MeshLightVisibility = uint8_t{1} << 3u;
static constexpr uint8_t SceneVisibility = MeshVisibility | GaussianVisibility;
static constexpr uint8_t PathVisibility = SceneVisibility
    | AnalyticLightVisibility | MeshLightVisibility;
static constexpr uint8_t FullVisibility = ~uint8_t{0};
// The path SBT keeps mesh at record 0 and Gaussian at record 1.  This is
// intentionally shared by the raygen SER classifier and the runtime SBT
// builder, so Gaussian/mesh classification never depends on triangle IDs.
static constexpr uint32_t PathGaussianSbtRecord = 1u;
static constexpr uint32_t PathAnalyticLightSbtRecord = 2u;
// The combined path SBT appends traversal-only query records after the three
// shading records: query mesh = 3, query Gaussian = 4.
static constexpr uint32_t QuerySbtRecordOffset = 3u;

using TlasHandle = uint64_t;

// POD result of an OptiX scene query. Mesh hits carry barycentrics; Gaussian
// hits leave primitiveIndex invalid and use instanceIndex as their Gaussian id.
struct RayHit
{
    float t{Ray::InfiniteDistance};
    float u{};
    float v{};
    uint32_t instanceIndex{InvalidIndex};
    uint32_t primitiveIndex{InvalidIndex};

    NR_CPU_GPU bool isValid(const float tMin, const float tMax) const
    {
        return instanceIndex != InvalidIndex
            && nr::isFinite(t) && t >= tMin && t <= tMax;
    }

};

struct CameraSample
{
    Ray ray{};
    float weight{1.0f};
};

struct PathRandomStreams
{
    RandomState opacity{};
    RandomState bsdf{};
    RandomState light{};
    RandomState shadow{};
    RandomState roulette{};
};

struct alignas(16) PathState
{
    SampledSpectrum throughput;
    SampledSpectrum radiance;
    SampledWavelengths wl;
    RandomState rngState;
    uint32_t depth;
    float alpha;
    float lastBsdfPdf;
    float etaScale;

    NR_CPU_GPU PathRandomStreams nextRandomStreams(
        const uint32_t pixel, const uint32_t accumulatedSamples)
    {
        const RandomState bounceKey = depth == 0
            ? seedRandom((static_cast<uint64_t>(accumulatedSamples) << 32u)
                | static_cast<uint64_t>(pixel))
            : rngState;
        rngState = advanceRandomSequence(bounceKey);
        return {
            forkRandom(bounceKey, RandomStream::Opacity),
            forkRandom(bounceKey, RandomStream::Bsdf),
            forkRandom(bounceKey, RandomStream::Light),
            forkRandom(bounceKey, RandomStream::Shadow),
            forkRandom(bounceKey, RandomStream::Roulette)};
    }

    NR_CPU_GPU void scatter(const SampledSpectrum& weight, const float pdf)
    {
        throughput *= weight;
        lastBsdfPdf = pdf;
        ++depth;
    }

    NR_CPU_GPU void transmit(const float eta)
    {
        etaScale *= eta * eta;
    }

};

static_assert(sizeof(PathState) == (NrSpectrumSamples == 1 ? 48 : 96));

// Beauty-path OptiX traces pass a pointer to this state through two payload
// registers.  The closest-hit and miss programs update the path in place,
// leaving ray generation responsible only for launching the next segment.
enum class PathTraceStatus : uint32_t
{
    Continue,
    Terminate,
};

struct PathTracePayload
{
    PathState* state{};
    Ray ray{};
    uint32_t pixel{};
    uint32_t gaussianSampleIndex{};
    float gaussianTMax{Ray::DefaultMaxDistance};
    PathTraceStatus status{PathTraceStatus::Terminate};
    // Regular beauty paths already intersect emissive mesh triangles through
    // the scene TLAS. Mesh entries in the combined light GAS are reserved for
    // dedicated light-hit queries, so keeping this false prevents coincident
    // mesh/light-GAS intersections from competing in the path traversal.
    uint32_t includeMeshLightHits{};
};

// Keep this distinct from OpaqueAovQueryMarker. Both values travel through
// OptiX payload 2, and confusing an AOV scalar payload for a path-payload
// pointer turns the next any-hit callback into an invalid dereference.
inline constexpr uint32_t PathTracePayloadMarker = 0xFFFFFFFCu;

NR_GPU inline uint32_t packPathPointer0(const void* pointer)
{
    return static_cast<uint32_t>(reinterpret_cast<uint64_t>(pointer) >> 32u);
}

NR_GPU inline uint32_t packPathPointer1(const void* pointer)
{
    return static_cast<uint32_t>(reinterpret_cast<uint64_t>(pointer));
}

#if defined(NR_GPU_CODE)
template <typename T = PathTracePayload>
NR_GPU inline T* getPathTracePayload()
{
    const uint64_t pointer = (static_cast<uint64_t>(optixGetPayload_0()) << 32u)
        | optixGetPayload_1();
    return reinterpret_cast<T*>(pointer);
}
#endif
