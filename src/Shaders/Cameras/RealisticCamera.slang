#ifndef REALISTIC_CAMERA_SLANG
#define REALISTIC_CAMERA_SLANG

#include "ICamera.slang"
#ifndef RAY_LUT_GENERATION_SHADER
#include "ThinLensCamera.slang"
#endif

bool intersectRossSphere(RealisticLensElement element, float surfaceOffset, LocalRay ray, out float t, out float3 normal)
{
    t = 0.0;
    normal = float3(0.0, 0.0, 1.0);

    if (abs(element.radius) <= 1e-7) {
        if (abs(ray.direction.z) <= 1e-7)
            return false;
        float surfaceCenter = element.vertexZ + surfaceOffset;
        t = (-surfaceCenter - ray.origin.z) / ray.direction.z;
        normal = ray.direction.z > 0.0 ? float3(0.0, 0.0, -1.0) : float3(0.0, 0.0, 1.0);
        return t >= 0.0;
    }

    float surfaceCenter = element.vertexZ + surfaceOffset;
    float zCenter = -surfaceCenter + element.radius;
    float3 o = ray.origin - float3(0.0, 0.0, zCenter);
    float a = dot(ray.direction, ray.direction);
    float b = 2.0 * dot(ray.direction, o);
    float c = dot(o, o) - element.radius * element.radius;
    float discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0)
        return false;

    float root = sqrt(discriminant);
    float t0 = (-b - root) / (2.0 * a);
    float t1 = (-b + root) / (2.0 * a);
    bool useCloserT = (ray.direction.z > 0.0) != (element.radius < 0.0);
    t = useCloserT ? min(t0, t1) : max(t0, t1);
    if (t < 0.0)
        return false;

    normal = normalize(o + ray.direction * t);
    if (dot(normal, -ray.direction) < 0.0)
        normal = -normal;
    return true;
}

bool refractRoss(float3 point, float3 normalIn, LocalRay ray, float iorIn, float iorOut, out LocalRay result)
{
    result.origin = point;
    result.direction = ray.direction;

    float3 n = -normalIn;
    float cosThetaI = dot(n, ray.direction);
    float eta = iorIn / max(iorOut, 1e-7);
    if (cosThetaI < 0.0) {
        eta = 1.0 / max(eta, 1e-7);
        cosThetaI = -cosThetaI;
        n = -n;
    }

    float sin2ThetaI = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sin2ThetaT = sin2ThetaI / max(eta * eta, 1e-7);
    if (sin2ThetaT >= 1.0)
        return false;

    float cosThetaT = sqrt(1.0 - sin2ThetaT);
    float3 f = -ray.direction / eta;
    float3 f1 = n * (cosThetaI / eta - cosThetaT);
    result.direction = normalize(f + f1);
    return true;
}

bool traceFromFilmRoss(RealisticCameraSettings lens, LocalRay inputRay, out LocalRay outputRay)
{
    outputRay = inputRay;
    if (lens.elementCount <= 0)
        return false;

    // Public realistic-camera local space follows ROSS generation: film is at
    // z=0 and +Z points through the lens toward the scene. The sequential ROSS
    // intersection code flips Z internally, so surface centers are addressed as
    // -vertexZ and the final ray is flipped back before world mapping.
    LocalRay tracedRay;
    tracedRay.origin = float3(inputRay.origin.x, inputRay.origin.y, -inputRay.origin.z);
    tracedRay.direction = float3(inputRay.direction.x, inputRay.direction.y, -inputRay.direction.z);

    for (int i = lens.elementCount - 1; i >= 0; --i) {
        RealisticLensElement element = lens.elements[i];
        float surfaceCenter = element.vertexZ + lens.surfaceOffset;
        if (element.isAperture != 0) {
            if (abs(tracedRay.direction.z) <= 1e-7)
                return false;
            float t = (-surfaceCenter - tracedRay.origin.z) / tracedRay.direction.z;
            if (t < 0.0)
                return false;
            float3 p = tracedRay.origin + tracedRay.direction * t;
            if (dot(p.xy, p.xy) > element.apertureRadius * element.apertureRadius)
                return false;
            continue;
        }

        float t;
        float3 normal;
        if (!intersectRossSphere(element, lens.surfaceOffset, tracedRay, t, normal))
            return false;

        float3 point = tracedRay.origin + tracedRay.direction * t;
        if (dot(point.xy, point.xy) > element.apertureRadius * element.apertureRadius)
            return false;

        float etaI = element.ior;
        float etaT = (i > 0 && lens.elements[i - 1].ior != 0.0) ? lens.elements[i - 1].ior : 1.0;

        LocalRay refracted;
        if (!refractRoss(point, normal, tracedRay, etaT, etaI, refracted))
            return false;

        tracedRay.origin = refracted.origin;
        tracedRay.direction = -refracted.direction;
    }

    outputRay.origin = float3(tracedRay.origin.x, tracedRay.origin.y, -tracedRay.origin.z);
    outputRay.direction = float3(tracedRay.direction.x, tracedRay.direction.y, -tracedRay.direction.z);
    return true;
}

float2 sampleExitPupil(RealisticCameraSettings lens, float2 filmPos, float2 u)
{
    float rFilm = length(filmPos);
    int boundIndex = int((rFilm / max(lens.filmDiagonal * 0.5, 1e-7)) * float(lens.pupilBoundCount));
    boundIndex = clamp(boundIndex, 0, max(lens.pupilBoundCount - 1, 0));
    RealisticPupilBound bound = lens.pupilBounds[boundIndex];

    float2 p = lerp(bound.minBounds, bound.maxBounds, u);
    float sinTheta = rFilm > 0.0 ? filmPos.y / rFilm : 0.0;
    float cosTheta = rFilm > 0.0 ? filmPos.x / rFilm : 1.0;
    return float2(cosTheta * p.x - sinTheta * p.y,
                  sinTheta * p.x + cosTheta * p.y);
}

bool generateRealisticLocalRay(
    int2 pixelCoord,
    int2 screenSize,
    float2 rasterSample,
    float2 lensSample,
    out LocalRay localRay)
{
    RealisticCameraSettings lens = sceneSettings.realisticCamera;
    localRay.origin = float3(0.0);
    localRay.direction = float3(0.0, 0.0, 1.0);
    if (lens.elementCount <= 0 || lens.pupilBoundCount <= 0)
        return false;

    float2 uv = (float2(pixelCoord) + rasterSample) / float2(screenSize);
    float2 sensor = float2(uv.x - 0.5, 0.5 - uv.y) * float2(lens.sensorWidth, lens.sensorHeight);
    // ROSS/PBRT negate film X relative to raster/sensor X.
    float2 filmPos = float2(-sensor.x, -sensor.y);
    float2 pupilPos = sampleExitPupil(lens, filmPos, lensSample);

    LocalRay filmRay;
    filmRay.origin = float3(filmPos, 0.0);
    filmRay.direction = normalize(float3(pupilPos - filmPos, lens.rearElementZ));

    if (!traceFromFilmRoss(lens, filmRay, localRay))
        localRay = filmRay;
    return true;
}

#ifndef RAY_LUT_GENERATION_SHADER
struct RealisticCamera<TSampler> : ICamera<TSampler>
    where TSampler : ISampler
{
    void generateRay(
        int2 pixelCoord,
        int2 screenSize,
        inout SamplerState samplerState,
        inout TSampler sampler,
        bool deterministicSample,
        out float3 rayOrigin,
        out float3 rayDirection)
    {
        RealisticCameraSettings lens = sceneSettings.realisticCamera;
        if (lens.elementCount <= 0 || lens.pupilBoundCount <= 0) {
            ThinLensCamera<TSampler> fallbackCamera;
            fallbackCamera.generateRay(pixelCoord, screenSize, samplerState, sampler, deterministicSample, rayOrigin, rayDirection);
            return;
        }

        float2 rasterSample = deterministicSample ? float2(0.0) : (sampler.get2D(samplerState) - 0.5);
        float2 lensSample = deterministicSample ? float2(0.5) : sampler.get2D(samplerState);
        LocalRay tracedRay;
        if (!generateRealisticLocalRay(pixelCoord, screenSize, rasterSample, lensSample, tracedRay)) {
            ThinLensCamera<TSampler> fallbackCamera;
            fallbackCamera.generateRay(pixelCoord, screenSize, samplerState, sampler, deterministicSample, rayOrigin, rayDirection);
            return;
        }
        transformCameraLocalRayToWorld(tracedRay, rayOrigin, rayDirection);
    }
};
#endif

#endif // REALISTIC_CAMERA_SLANG
