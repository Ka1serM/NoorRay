#include "Kernels/cuda/CudaRayLutGenerator.h"

#include <cuda_runtime.h>

#include "Scene/SceneTypes.h"

namespace
{
using V2 = glm::vec2;
using V3 = glm::vec3;

__device__ bool intersectLensElement(
    const RealisticLensElement& elem, float surfaceOffset,
    V3 ro, V3 rd, float& t, V3& n)
{
    t = 0.0f; n = V3(0.0f, 0.0f, 1.0f);

    const float centerZ = elem.vertexZ + surfaceOffset;

    if (fabsf(elem.radius) <= 1e-7f)
    {
        if (fabsf(rd.z) <= 1e-7f) return false;
        t = (-centerZ - ro.z) / rd.z;
        n = rd.z > 0.0f ? V3(0.0f, 0.0f, -1.0f) : V3(0.0f, 0.0f, 1.0f);
        return t >= 0.0f;
    }

    const float zCenter = -centerZ + elem.radius;
    const V3 o = ro - V3(0.0f, 0.0f, zCenter);
    const float a = glm::dot(rd, rd);
    const float b = 2.0f * glm::dot(rd, o);
    const float c = glm::dot(o, o) - elem.radius * elem.radius;
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return false;

    const float root = sqrtf(disc);
    const float t0 = (-b - root) / (2.0f * a);
    const float t1 = (-b + root) / (2.0f * a);
    const bool useSmaller = (rd.z > 0.0f) != (elem.radius < 0.0f);
    t = useSmaller ? fminf(t0, t1) : fmaxf(t0, t1);
    if (t < 0.0f) return false;

    n = glm::normalize(o + rd * t);
    if (glm::dot(n, -rd) < 0.0f) n = -n;
    return true;
}

__device__ bool refractLens(V3 pt, V3 normalIn, V3 rd, float iorIn, float iorOut, V3& outOrigin, V3& outDir)
{
    outOrigin = pt; outDir = rd;
    V3 nn = -normalIn;
    float cosI = glm::dot(nn, rd);
    float eta = iorIn / fmaxf(iorOut, 1e-7f);
    if (cosI < 0.0f) { eta = 1.0f / fmaxf(eta, 1e-7f); cosI = -cosI; nn = -nn; }
    const float sin2I = fmaxf(0.0f, 1.0f - cosI * cosI);
    const float sin2T = sin2I / fmaxf(eta * eta, 1e-7f);
    if (sin2T >= 1.0f) return false;
    const float cosT = sqrtf(1.0f - sin2T);
    outDir = glm::normalize(-rd / eta + nn * (cosI / eta - cosT));
    return true;
}

__device__ bool traceFromFilm(
    const RealisticCameraSettings& lens,
    V3 ro, V3 rd,
    V3& outOrigin, V3& outDir)
{
    outOrigin = ro; outDir = rd;
    if (lens.elementCount <= 0) return false;

    // Film is at z=0, +Z through lens. ROSS internal convention flips Z.
    V3 tracedO = V3(ro.x, ro.y, -ro.z);
    V3 tracedD = V3(rd.x, rd.y, -rd.z);

    for (int i = lens.elementCount - 1; i >= 0; --i)
    {
        const RealisticLensElement& elem = lens.elements[i];
        const float surfaceCenter = elem.vertexZ + lens.surfaceOffset;

        if (elem.isAperture != 0)
        {
            if (fabsf(tracedD.z) <= 1e-7f) return false;
            const float t = (-surfaceCenter - tracedO.z) / tracedD.z;
            if (t < 0.0f) return false;
            const V3 p = tracedO + tracedD * t;
            if (p.x * p.x + p.y * p.y > elem.apertureRadius * elem.apertureRadius) return false;
            continue;
        }

        float t; V3 n;
        if (!intersectLensElement(elem, lens.surfaceOffset, tracedO, tracedD, t, n)) return false;
        const V3 pt = tracedO + tracedD * t;
        if (pt.x * pt.x + pt.y * pt.y > elem.apertureRadius * elem.apertureRadius) return false;

        const float etaI = elem.ior;
        const float etaT = (i > 0 && lens.elements[i - 1].ior != 0.0f)
            ? lens.elements[i - 1].ior : 1.0f;

        V3 newO, newD;
        if (!refractLens(pt, n, tracedD, etaT, etaI, newO, newD)) return false;
        tracedO = newO;
        tracedD = -newD;
    }

    outOrigin = V3(tracedO.x, tracedO.y, -tracedO.z);
    outDir    = V3(tracedD.x, tracedD.y, -tracedD.z);
    return true;
}

__device__ V2 sampleExitPupil(const RealisticCameraSettings& lens, V2 filmPos, V2 u)
{
    const float rFilm = glm::length(filmPos);
    const float halfDiag = fmaxf(lens.filmDiagonal * 0.5f, 1e-7f);
    int bi = static_cast<int>((rFilm / halfDiag) * static_cast<float>(lens.pupilBoundCount));
    bi = max(0, min(bi, max(lens.pupilBoundCount - 1, 0)));
    const RealisticPupilBound& bound = lens.pupilBounds[bi];
    const V2 p = bound.minBounds + (bound.maxBounds - bound.minBounds) * u;
    const float sinTh = rFilm > 0.0f ? filmPos.y / rFilm : 0.0f;
    const float cosTh = rFilm > 0.0f ? filmPos.x / rFilm : 1.0f;
    return V2(cosTh * p.x - sinTh * p.y, sinTh * p.x + cosTh * p.y);
}

__global__ void rayLutKernel(
    RealisticCameraSettings lens,
    RayLutEntry* output,
    uint32_t width,
    uint32_t height)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;

    const uint32_t px = idx % width;
    const uint32_t py = idx / width;

    RayLutEntry entry{};
    entry.direction = vec3(0.0f, 0.0f, 1.0f);
    entry.originValid = 0.0f;
    if (lens.elementCount <= 0 || lens.pupilBoundCount <= 0)
    {
        output[idx] = entry;
        return;
    }

    // Center-pixel sample, center exit-pupil sample (deterministic LUT)
    const V2 uv = (V2(px, py) + V2(0.5f)) / V2(width, height);
    const V2 sensor = (V2(uv.x - 0.5f, 0.5f - uv.y)) * V2(lens.sensorWidth, lens.sensorHeight);
    const V2 filmPos = V2(-sensor.x, -sensor.y);
    const V2 pupilPos = sampleExitPupil(lens, filmPos, V2(0.5f));

    const V3 filmOrigin = V3(filmPos, 0.0f);
    const V3 filmDir = glm::normalize(V3(pupilPos - filmPos, lens.rearElementZ));

    V3 outO, outD;
    if (traceFromFilm(lens, filmOrigin, filmDir, outO, outD))
    {
        entry.origin = outO;
        entry.direction = glm::normalize(outD);
        entry.originValid = 1.0f;
    }

    output[idx] = entry;
}
}

void generateCudaRayLut(
    const RealisticCameraSettings& settings,
    RayLutEntry* deviceOutput,
    const uint32_t width,
    const uint32_t height,
    const cudaStream_t stream)
{
    constexpr uint32_t blockSize = 256;
    const uint32_t count = width * height;
    rayLutKernel<<<(count + blockSize - 1) / blockSize, blockSize, 0, stream>>>(
        settings, deviceOutput, width, height);
}
