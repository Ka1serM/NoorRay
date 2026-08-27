#include "RaytracingAbi.slang"
#include "Environment.slang"
#include "RgbToSpectrum.slang"
#include "Sampling.slang"
#include "CameraLens.slang"
#include "AnalyticLights.slang"

// Raygen owns only camera-ray construction, traversal, and output writes.
// Surface decoding, MaterialX SVM, Gaussian acceptance, and Gaussian shading
// live in hit stages so the dispatch shader stays small and predictable.
RaytracingAccelerationStructure accelerationStructures[];

float cameraFloat(ByteAddressBuffer camera, uint byteOffset)
{
    return asfloat(camera.Load(byteOffset));
}

float3 cameraTransformPoint(ByteAddressBuffer camera, float3 p)
{
    return float3(
        dot(float4(p, 1.0), float4(cameraFloat(camera, 0),
            cameraFloat(camera, 4), cameraFloat(camera, 8),
            cameraFloat(camera, 12))),
        dot(float4(p, 1.0), float4(cameraFloat(camera, 16),
            cameraFloat(camera, 20), cameraFloat(camera, 24),
            cameraFloat(camera, 28))),
        dot(float4(p, 1.0), float4(cameraFloat(camera, 32),
            cameraFloat(camera, 36), cameraFloat(camera, 40),
            cameraFloat(camera, 44))));
}

float3 cameraTransformVector(ByteAddressBuffer camera, float3 v)
{
    return normalize(float3(
        dot(v, float3(cameraFloat(camera, 0), cameraFloat(camera, 4),
            cameraFloat(camera, 8))),
        dot(v, float3(cameraFloat(camera, 16), cameraFloat(camera, 20),
            cameraFloat(camera, 24))),
        dot(v, float3(cameraFloat(camera, 32), cameraFloat(camera, 36),
            cameraFloat(camera, 40)))));
}

[shader("raygeneration")]
void main()
{
    const FrameParams frame = *root.arguments;
    const uint3 dispatchId = DispatchRaysIndex();
    if (dispatchId.x >= frame.width || dispatchId.y >= frame.height)
        return;

    RWTexture2D<float4> color = DescriptorHandle<RWTexture2D<float4>>(
        uint2(frame.colorImage, 0));
    RWTexture2D<float4> albedo = DescriptorHandle<RWTexture2D<float4>>(
        uint2(frame.albedoImage, 0));
    RWTexture2D<float4> normal = DescriptorHandle<RWTexture2D<float4>>(
        uint2(frame.normalImage, 0));
    RWTexture2D<float4> position = DescriptorHandle<RWTexture2D<float4>>(
        uint2(frame.positionImage, 0));
    RWTexture2D<uint> cryptomatte = DescriptorHandle<RWTexture2D<uint>>(
        uint2(frame.cryptomatteImage, 0));
    RWStructuredBuffer<uint> gaussianOverdraw = DescriptorHandle<RWStructuredBuffer<uint>>(
        uint2(frame.gaussianOverdrawImage, 0));
    RWStructuredBuffer<float4> accumulation =
        DescriptorHandle<RWStructuredBuffer<float4>>(
            uint2(frame.accumulationBuffer, 0));
    ByteAddressBuffer camera = DescriptorHandle<ByteAddressBuffer>(
        uint2(uint(frame.cameraAddress), 0));
    ByteAddressBuffer spectralTables = DescriptorHandle<ByteAddressBuffer>(
        uint2(frame.spectralTables, 0));

    const uint pixel = dispatchId.y * frame.width + dispatchId.x;
    gaussianOverdraw[pixel] = 0u;
    // Proxy overdraw is a diagnostic, not a beauty sample. Fix its primary
    // ray at the pixel and lens centres so the displayed counter is stable
    // across accumulation frames and independent of path-sampling noise.
    const bool proxyOverdraw = frame.gaussianOverdrawEnabled != 0u;
    PathSampler primarySampler;
    primarySampler.sampleIndex = frame.sampleIndex;
    primarySampler.scramble = hashCombine32(frame.frameIndex, pixel);
    Rng pathRng;
    pathRng.state = pcgHash(primarySampler.scramble
        ^ (frame.sampleIndex * 0x9E3779B9u));
    const float2 pixelSample = proxyOverdraw || frame.aovEnabled != 0u || frame.sampleIndex == 0u
        ? float2(0.5) : primarySampler.next2D(0u);
    const float2 uv = (float2(dispatchId.xy) + pixelSample)
        / float2(frame.width, frame.height);
    const float sensorWidth = max(cameraFloat(camera, 68u), 0.001);
    const float sensorHeight = max(cameraFloat(camera, 72u), 0.001);
    const float focalLength = max(cameraFloat(camera, 76u), 0.001);
    const float wavelengthSample = primarySampler.next1D(2u);
    SampledWavelengths wavelengths = sampleWavelengthsVisible(
        wavelengthSample);
    const uint projection = camera.Load(64u);
    const float focusDistance = max(cameraFloat(camera, 80u), 0.1) * 0.01;
    const float apertureRadius = max(cameraFloat(camera, 84u), 0.0) * 0.0005;
    float2 normalizedSensor = (uv - 0.5) * 2.0;
    // Presentation flips the storage image back to NoorRay's historical
    // bottom-left GPU convention. Upper-left therefore already has positive
    // film Y at the displayed top; lower-left reverses that camera mapping.
    if (camera.Load(88u) != 0u)
        normalizedSensor.y = -normalizedSensor.y;
    float3 localOrigin = float3(0.0);
    float3 localDirection;
    if (projection == 1u) // Orthographic
    {
        const float height = 10.0 * 21.0 / focalLength;
        localOrigin = float3(normalizedSensor.x * height
            * sensorWidth / sensorHeight * 0.5,
            normalizedSensor.y * height * 0.5, 0.0);
        localDirection = float3(0.0, 0.0, -1.0);
    }
    else if (projection == 2u) // Fisheye
    {
        const float aspect = sensorWidth / sensorHeight;
        const float2 p = float2(normalizedSensor.x * aspect,
            normalizedSensor.y);
        const float radius = length(p);
        if (radius < 1.0e-6)
            localDirection = float3(0.0, 0.0, -1.0);
        else
        {
            const float radiusMax = sqrt(aspect * aspect + 1.0);
            const float halfFov = atan(sensorWidth / (2.0 * focalLength));
            const float theta = radius / radiusMax * halfFov;
            localDirection = float3(p * (sin(theta) / radius), -cos(theta));
        }
    }
    else // Perspective and thin lens
    {
        localDirection = normalize(float3(
            normalizedSensor.x * sensorWidth / (2.0 * focalLength),
            normalizedSensor.y * sensorHeight / (2.0 * focalLength), -1.0));
    }
    bool cameraRayValid = true;
    float cameraWeight = 1.0;
    if (projection == 4u) // Realistic sequential lens
    {
        ByteAddressBuffer lens = DescriptorHandle<ByteAddressBuffer>(
            uint2(uint(frame.lensAddress), 0));
        const uint surfaceCount = lens.Load(LensSurfaceCountOffset);
        const float pupilZ = lensFloat(lens, LensRearPupilZOffset);
        const float pupilRadius = lensFloat(lens, LensRearPupilRadiusOffset);
        if (surfaceCount == 0u || pupilRadius <= 0.0)
            cameraRayValid = false;
        else
        {
            const float2 square = proxyOverdraw ? float2(0.0)
                : primarySampler.next2D(3u) * 2.0 - 1.0;
            float pupilR;
            float pupilPhi;
            if (abs(square.x) > abs(square.y))
            {
                pupilR = square.x;
                pupilPhi = 0.785398163 * square.y
                    / (square.x == 0.0 ? 1.0 : square.x);
            }
            else
            {
                pupilR = square.y;
                pupilPhi = 1.570796327 - 0.785398163 * square.x
                    / (square.y == 0.0 ? 1.0 : square.y);
            }
            const float2 pupil = pupilRadius * pupilR
                * float2(cos(pupilPhi), sin(pupilPhi));
            float3 lensOrigin = float3(-normalizedSensor.x * sensorWidth * 0.5,
                -normalizedSensor.y * sensorHeight * 0.5, 0.0);
            float3 lensDirection = normalize(float3(pupil, pupilZ) - lensOrigin);
            wavelengths.terminateSecondary();
            cameraRayValid = traceLensFilmToWorld(lens, lensOrigin,
                lensDirection, wavelengths.lambda.x);
            if (cameraRayValid)
            {
                const float cosineSquared = lensDirection.z * lensDirection.z;
                cameraWeight = 3.141592654 * pupilRadius * pupilRadius
                    * cosineSquared * cosineSquared
                    / max(1.0e-6, pupilZ * pupilZ);
                localOrigin = lensOrigin * 0.001;
                localDirection = normalize(float3(lensDirection.xy,
                    -lensDirection.z));
            }
        }
    }
    else if ((projection == 2u || projection == 3u)
        && apertureRadius > 0.0)
    {
        const float2 lensSample = proxyOverdraw ? float2(0.5)
            : primarySampler.next2D(3u);
        const float angle = 6.28318530718 * lensSample.x;
        const float radius = apertureRadius * sqrt(lensSample.y);
        localOrigin = float3(cos(angle) * radius, sin(angle) * radius, 0.0);
        const float3 focusPoint = localDirection
            * (focusDistance / max(-localDirection.z, 1.0e-5));
        localDirection = normalize(focusPoint - localOrigin);
    }

    float3 rayOrigin = cameraTransformPoint(camera, localOrigin);
    float3 rayDirection = cameraTransformVector(camera, localDirection);
    SampledSpectrum throughput = SampledSpectrum(cameraWeight);
    SampledSpectrum pathRadiance = SampledSpectrum(0.0);
    float3 firstAlbedo = float3(0.0);
    float3 firstNormal = float3(0.0, 0.0, 1.0);
    float3 firstPosition = float3(0.0);
    uint firstCryptomatte = 0xFFFFFFFFu;
    float alpha = 1.0;
    bool recordedFirstHit = false;
    float lastBsdfPdf = 0.0;

    // No indirect rays belong in a screen-space proxy-overdraw measurement.
    const uint bounceCount = cameraRayValid
        ? (proxyOverdraw ? 1u : max(frame.maxBounces, 1u)) : 0u;
    [loop]
    for (uint depth = 0u; depth < bounceCount; ++depth)
    {
        RayDesc ray;
        ray.Origin = rayOrigin;
        ray.Direction = rayDirection;
        ray.TMin = 0.001;
        ray.TMax = 1.0e30;

        RayPayload payload = {};
        payload.gaussianId = 0xFFFFFFFFu;
        payload.seed = pathRng.state;
        payload.wavelengthSample = wavelengthSample;
        payload.depth = depth;
        if (frame.topLevelAS != 0xFFFFFFFFu)
            TraceRay(accelerationStructures[frame.topLevelAS], RAY_FLAG_NONE,
                0xFF, 0, 2, 0, ray, payload);
        pathRng.state = payload.seed;

        if (payload.hit == 0u)
        {
            if (frame.environmentBuffer != 0xFFFFFFFFu)
            {
                StructuredBuffer<EnvironmentRecord> environmentBuffer =
                    DescriptorHandle<StructuredBuffer<EnvironmentRecord>>(
                        uint2(frame.environmentBuffer, 0));
                const float3 environment = environmentRadianceRgb(
                    environmentBuffer[0], rayDirection, depth == 0u);
                float misWeight = 1.0;
                if (depth != 0u && lastBsdfPdf > 0.0
                    && uint(frame.sceneAddress) != 0xFFFFFFFFu)
                {
                    ByteAddressBuffer lights = DescriptorHandle<ByteAddressBuffer>(
                        uint2(uint(frame.sceneAddress), 0));
                    const float finiteWeight = max(lightFloat(lights, 4u), 0.0);
                    const float environmentWeight = max(
                        environmentBuffer[0].importanceWeight, 0.0);
                    const float totalWeight = finiteWeight + environmentWeight;
                    const float environmentProbability = totalWeight > 0.0
                        ? environmentWeight / totalWeight : 0.0;
                    const float lightPdf = environmentProbability
                        * environmentPdf(environmentBuffer[0], rayDirection);
                    if (lightPdf > 0.0)
                        misWeight = powerHeuristic(lastBsdfPdf, lightPdf);
                }
                if (depth != 0u || frame.transparentBackground == 0u)
                {
                    const SampledSpectrum contribution = throughput
                        * rgbIlluminantToSpectrum(spectralTables, environment,
                            wavelengths) * misWeight;
                    pathRadiance += clampIndirectContribution(contribution,
                        frame.indirectLightClamp, depth);
                }
                else
                    alpha = 0.0;
                if (!recordedFirstHit)
                    firstAlbedo = environment;
            }
            break;
        }

        if (payload.hit == 4u)
        {
            float misWeight = 1.0;
            if (depth != 0u && lastBsdfPdf > 0.0 && payload.lightPdf > 0.0
                && frame.environmentBuffer != 0xFFFFFFFFu)
            {
                ByteAddressBuffer lights = DescriptorHandle<ByteAddressBuffer>(
                    uint2(uint(frame.sceneAddress), 0));
                StructuredBuffer<EnvironmentRecord> environments =
                    DescriptorHandle<StructuredBuffer<EnvironmentRecord>>(
                        uint2(frame.environmentBuffer, 0));
                const float finiteWeight = max(lightFloat(lights, 4u), 0.0);
                const float environmentWeight = max(
                    environments[0].importanceWeight, 0.0);
                const float totalWeight = finiteWeight + environmentWeight;
                const float finiteProbability = totalWeight > 0.0
                    ? finiteWeight / totalWeight : 0.0;
                misWeight = powerHeuristic(lastBsdfPdf,
                    finiteProbability * payload.lightPdf);
            }
            pathRadiance += clampIndirectContribution(throughput
                * payload.emissionSpectrum * misWeight,
                frame.indirectLightClamp, depth);
            if (!recordedFirstHit)
            {
                recordedFirstHit = true;
                firstPosition = payload.position;
                firstCryptomatte = 0xFFFFFFFFu;
            }
            break;
        }

        if (!recordedFirstHit)
        {
            recordedFirstHit = true;
            firstAlbedo = payload.albedo;
            firstNormal = payload.normal;
            firstPosition = payload.position;
            firstCryptomatte = payload.cryptomatte;
            alpha = payload.alpha;
        }
        if (payload.hit == 2u)
        {
            pathRadiance += clampIndirectContribution(throughput
                * rgbIlluminantToSpectrum(spectralTables, payload.radiance,
                    wavelengths), frame.indirectLightClamp, depth);
            break;
        }

        float emissionMisWeight = 1.0;
        if (depth != 0u && payload.hit == 1u && lastBsdfPdf > 0.0
            && uint(frame.sceneAddress) != 0xFFFFFFFFu
            && frame.environmentBuffer != 0xFFFFFFFFu)
        {
            ByteAddressBuffer lights = DescriptorHandle<ByteAddressBuffer>(
                uint2(uint(frame.sceneAddress), 0));
            StructuredBuffer<EnvironmentRecord> environments =
                DescriptorHandle<StructuredBuffer<EnvironmentRecord>>(
                    uint2(frame.environmentBuffer, 0));
            const float finiteWeight = max(lightFloat(lights, 4u), 0.0);
            const float environmentWeight = max(environments[0].importanceWeight, 0.0);
            const float totalWeight = finiteWeight + environmentWeight;
            const float finiteProbability = totalWeight > 0.0
                ? finiteWeight / totalWeight : 0.0;
            const float lightPdf = finiteProbability * meshLightHitPdf(lights,
                payload.instance, payload.primitive, rayOrigin, payload.position);
            if (lightPdf > 0.0)
                emissionMisWeight = powerHeuristic(lastBsdfPdf, lightPdf);
        }
        pathRadiance += clampIndirectContribution(throughput
            * payload.emissionSpectrum * emissionMisWeight,
            frame.indirectLightClamp, depth);
        if (payload.shadowValid != 0u)
        {
            RayDesc shadowRay;
            const float shadowOffsetSign = dot(payload.normal,
                payload.shadowDirection) >= 0.0 ? 1.0 : -1.0;
            shadowRay.Origin = payload.position + payload.normal
                * (shadowOffsetSign * 1.0e-4);
            shadowRay.Direction = normalize(payload.shadowDirection);
            shadowRay.TMin = 0.001;
            shadowRay.TMax = payload.shadowDistance < 1.0e15
                ? max(payload.shadowDistance + (payload.shadowValid == 2u
                    ? 2.0e-4 : -2.0e-4), shadowRay.TMin)
                : 1.0e30;
            RayPayload shadowPayload = {};
            shadowPayload.hit = payload.shadowValid == 2u ? 0u : 1u;
            shadowPayload.instance = payload.shadowInstance;
            shadowPayload.primitive = payload.shadowPrimitive;
            shadowPayload.gaussianId = payload.hit == 3u
                ? payload.gaussianId : 0xFFFFFFFFu;
            shadowPayload.seed = pathRng.state;
            shadowPayload.wavelengthSample = wavelengthSample;
            shadowPayload.depth = depth;
            shadowPayload.shadowValid = payload.shadowValid;
            const uint rayType = payload.shadowValid == 2u ? 1u : 0u;
            const uint flags = payload.shadowValid == 2u
                ? RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
                : RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
                    | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
            TraceRay(accelerationStructures[frame.topLevelAS], flags,
                0xFF, rayType, 2, 0, shadowRay, shadowPayload);
            if (payload.shadowValid == 2u && shadowPayload.hit == 1u)
                pathRadiance += clampIndirectContribution(throughput
                    * payload.directSpectrum * shadowPayload.emissionSpectrum,
                    frame.indirectLightClamp, depth);
            else if (payload.shadowValid == 1u && shadowPayload.hit == 0u)
                pathRadiance += clampIndirectContribution(throughput
                    * payload.directSpectrum, frame.indirectLightClamp, depth);
        }
        if (payload.sampleValid == 0u)
            break;
        throughput *= payload.bsdfWeight;
        lastBsdfPdf = payload.bsdfPdf;
        if (!spectrumIsFinite(throughput)
            || spectrumMaxAbsComponent(throughput) <= 0.0)
            break;

        if (depth >= 3u)
        {
            const float survival = clamp(spectrumMaxComponent(throughput),
                0.05, 0.95);
            if (pathRng.nextFloat() >= survival)
                break;
            throughput *= 1.0 / survival;
        }
        rayDirection = normalize(payload.nextDirection);
        const float offsetSign = dot(payload.normal, rayDirection) >= 0.0
            ? 1.0 : -1.0;
        rayOrigin = payload.position + payload.normal * (offsetSign * 1.0e-4);
    }

    const float3 beauty = max(xyzToLinearSRGB(spectrumToXYZ(
        pathRadiance, wavelengths, spectralTables)), 0.0) * exp2(frame.exposure);
    // The sample count is already known from the push ABI, so the alpha slot
    // accumulates coverage instead. A single sample only ever reports full or
    // zero coverage; averaging it is what antialiases the transparent
    // background and the stochastic opacity resolved during traversal.
    float4 sum = frame.sampleIndex == 0u
        ? float4(0.0) : accumulation[pixel];
    sum.rgb += beauty;
    sum.a += alpha;
    accumulation[pixel] = sum;
    const float sampleCount = float(frame.sampleIndex + 1u);
    color[dispatchId.xy] = float4(sum.rgb / sampleCount,
        saturate(sum.a / sampleCount));
    albedo[dispatchId.xy] = float4(firstAlbedo, 1.0);
    normal[dispatchId.xy] = float4(firstNormal, 1.0);
    position[dispatchId.xy] = float4(firstPosition, 1.0);
    cryptomatte[dispatchId.xy] = firstCryptomatte;
}
