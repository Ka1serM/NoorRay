#ifndef RAYTRACING_INTEGRATORS_SLANG
#define RAYTRACING_INTEGRATORS_SLANG

static const float QUEUE_AO_MAX_DISTANCE = 1.5;
static const int   QUEUE_AO_SAMPLES      = 4;

float estimateQueueAmbientOcclusion(float3 worldPosition, float3 geometricNormal, inout uint rngState)
{
    float occluded = 0.0;
    for (int i = 0; i < QUEUE_AO_SAMPLES; ++i) {
        float3 dir   = sampleDiffuse(geometricNormal, rngState);
        float  nDotL = max(dot(geometricNormal, dir), 0.0);
        float  bias  = computeShadowBias(worldPosition, nDotL);
        if (traceShadowPath(worldPosition + geometricNormal * bias, dir, bias, QUEUE_AO_MAX_DISTANCE))
            occluded += 1.0;
    }
    return 1.0 - (occluded / float(QUEUE_AO_SAMPLES));
}

bool hiddenEnvironmentOutputsTransparent()
{
    return sceneSettings.environment.visible == 0 ||
        sceneSettings.renderSettings.transparentBackground != 0;
}

void clearPathVisibility(inout PathState state)
{
    setHitAnything(state, false);
    state.PrimaryObjectIndex = InvalidInstance;
    state.PrimaryPosition    = float3(0.0);
    state.PrimaryAlbedo      = float3(0.0);
    state.PrimaryNormal      = float3(0.0);
}

interface IPathIntegrator
{
    uint  sampleCount();
    void  shadeMiss(float3 worldRayDirection, inout PathState state);
    void  shadeSurface(SurfaceData surface, float3 worldRayDirection, uint sampleIndex, inout PathState state, inout Payload payload);
    float3 filterRadiance(PathState state, float3 referenceRadiance);
};

// ── Ambient occlusion integrator ──────────────────────────────────────────────

struct AoIntegrator : IPathIntegrator
{
    uint sampleCount() { return uint(max(sceneSettings.renderSettings.samples, 1)); }

    void shadeMiss(float3 worldRayDirection, inout PathState state)
    {
        uint   missFlags = uint(BounceType.RAY_TERMINATED);
        float3 envColor  = evaluateEnvironmentVisibleRadiance(worldRayDirection, sceneSettings.environment);
        if (sceneSettings.environment.visible == 0 || sceneSettings.renderSettings.transparentBackground != 0) {
            envColor = float3(0.0);
            missFlags |= uint(BounceType.ENV_TRANSPARENT);
        }

        state.Radiance += envColor;
        state.Flags     = missFlags;
        if (!getHitAnything(state)) {
            setHitAnything(state, (missFlags & uint(BounceType.ENV_TRANSPARENT)) == 0u);
            state.PrimaryObjectIndex = InvalidInstance;
            state.PrimaryAlbedo      = getHitAnything(state) ? envColor : float3(0.0);
            state.PrimaryNormal      = float3(0.0);
            state.PrimaryPosition    = float3(0.0);
        }
    }

    void shadeSurface(SurfaceData surface, float3 worldRayDirection, uint sampleIndex, inout PathState state, inout Payload payload)
    {
        float3 viewDir              = fastNormalize(-worldRayDirection);
        float3 geometricFacingNormal = computeFacingGeometricNormal(surface.geometricNormal, viewDir);

        if (rand(payload.RngState) > surface.opacity) {
            state.RngState     = payload.RngState;
            state.RayOrigin    = surface.worldPosition + worldRayDirection * computeRayOriginBias(surface.worldPosition);
            state.RayDirection = worldRayDirection;
            state.Flags        = uint(BounceType.RAY_TRANSPARENT);
            enqueueContinuationRay(sampleIndex, state);
            return;
        }

        float3 facingNormal = computeFacingShadingNormal(surface.shadingNormal, viewDir, geometricFacingNormal);
        float  ao           = estimateQueueAmbientOcclusion(surface.worldPosition, geometricFacingNormal, payload.RngState);
        float3 aoColor      = sceneSettings.renderSettings.aoSampleAlbedo != 0
            ? surface.albedo * ao : float3(ao);

        state.Radiance += aoColor;
        setHitAnything(state, true);
        state.PrimaryAlbedo      = surface.albedo;
        state.PrimaryNormal      = facingNormal * 0.5 + 0.5;
        state.PrimaryPosition    = surface.worldPosition;
        state.RngState           = payload.RngState;
        state.Flags              = uint(BounceType.RAY_TERMINATED);
        pathStates[sampleIndex]  = state;
    }

    float3 filterRadiance(PathState state, float3 referenceRadiance)
    {
        return max(state.Radiance, float3(0.0));
    }
};

// ── Path-tracing integrator ───────────────────────────────────────────────────

struct PathTracingIntegrator : IPathIntegrator
{
    uint sampleCount() { return 1u; }

    void shadeMiss(float3 worldRayDirection, inout PathState state)
    {
        uint   missFlags;
        uint   previousFlags = state.Flags;
        float3 missRadiance = evaluateMissRadiance(worldRayDirection, state.Depth, state.Flags, state.LastBsdfPdf, missFlags);

        state.Radiance += state.Throughput * missRadiance;
        state.Flags     = missFlags;
        if ((previousFlags & uint(BounceType.BOUNCE_TRANSMIT)) != 0u &&
            (missFlags & uint(BounceType.ENV_TRANSPARENT)) != 0u) {
            clearPathVisibility(state);
            return;
        }

        if (state.Depth == 0u && !getHitAnything(state)) {
            setHitAnything(state, (missFlags & uint(BounceType.ENV_TRANSPARENT)) == 0u);
            state.PrimaryObjectIndex = InvalidInstance;
            state.PrimaryPosition    = float3(0.0);
            state.PrimaryAlbedo      = getHitAnything(state) ? missRadiance : float3(0.0);
            state.PrimaryNormal      = float3(0.0);
        }
    }

    void shadeSurface(SurfaceData surface, float3 worldRayDirection, uint sampleIndex, inout PathState state, inout Payload payload)
    {
        float3 viewDir               = fastNormalize(-worldRayDirection);
        float3 geometricFacingNormal  = computeFacingGeometricNormal(surface.geometricNormal, viewDir);

        if (rand(payload.RngState) > surface.opacity) {
            state.RngState     = payload.RngState;
            state.RayOrigin    = surface.worldPosition + worldRayDirection * computeRayOriginBias(surface.worldPosition);
            state.RayDirection = worldRayDirection;
            state.Flags        = uint(BounceType.RAY_TRANSPARENT);
            enqueueContinuationRay(sampleIndex, state);
            return;
        }

        float3 albedo      = surface.albedo;
        float3 emission    = surface.emission;
        float  metallic    = surface.metallic;
        float  specular    = surface.specular;
        float  roughness   = regularizeSecondaryRoughness(surface.roughness, state.Depth);
        float  transmission = surface.transmission;
        float3 facingNormal = computeFacingShadingNormal(surface.shadingNormal, viewDir, geometricFacingNormal);

        if (state.Depth == 0u) {
            setHitAnything(state, true);
            state.PrimaryAlbedo   = albedo;
            state.PrimaryNormal   = facingNormal * 0.5 + 0.5;
            state.PrimaryPosition = surface.worldPosition;
        }

        state.Radiance += state.Throughput * emission;

        float transmissionProb = clamp(transmission, 0.0, 1.0);
        float opaqueProb       = max(1.0 - transmissionProb, EPSILON);

        if (rand(payload.RngState) < transmissionProb) {
            handleDielectricBSDF(viewDir, facingNormal, roughness, surface.ior, surface.transmissionColor, payload);
            payload.Attenuation /= max(transmissionProb, EPSILON);
        } else {
            float  dielectricF0 = dielectricF0FromSpecular(specular);
            float3 F0           = lerp(float3(dielectricF0), albedo, metallic);
            float  swSpecular   = computeSpecularSamplingWeight(viewDir, facingNormal, albedo, F0, metallic, roughness, specular);

            // ONE shadow ray per bounce — no concurrent write race in ConnectCompute.
            queueNEEShadowRay(
                sampleIndex, state.Throughput,
                surface.worldPosition, surface.worldShadowPosition,
                facingNormal, geometricFacingNormal,
                viewDir, albedo, specular, metallic, roughness, swSpecular,
                payload.RngState);

            handleOpaqueBSDF(viewDir, facingNormal, geometricFacingNormal, albedo, metallic, specular, roughness, payload);
            payload.Attenuation /= opaqueProb;
        }

        payload.NextDirection = fastNormalize(payload.NextDirection);
        if ((payload.Flags & uint(BounceType.BOUNCE_DIFFUSE))   != 0u) incrementDiffuseCount(state);
        if ((payload.Flags & uint(BounceType.BOUNCE_SPECULAR))  != 0u) incrementSpecularCount(state);
        if ((payload.Flags & uint(BounceType.BOUNCE_TRANSMIT))  != 0u) incrementTransmissionCount(state);

        state.Throughput  *= payload.Attenuation;
        state.RngState     = payload.RngState;
        state.LastBsdfPdf  = payload.LastBsdfPdf;
        state.Flags        = payload.Flags;

        if (getDiffuseCount(state)      > uint(sceneSettings.renderSettings.diffuseBounces)      ||
            getSpecularCount(state)     > uint(sceneSettings.renderSettings.specularBounces)     ||
            getTransmissionCount(state) > uint(sceneSettings.renderSettings.transmissionBounces)) {
            if ((payload.Flags & uint(BounceType.BOUNCE_TRANSMIT)) != 0u && hiddenEnvironmentOutputsTransparent())
                clearPathVisibility(state);
            pathStates[sampleIndex] = state;
            return;
        }

        float rrProb = computeRussianRouletteSurvivalProbability(
            state.Throughput, int(state.Depth), effectiveRussianRouletteStartBounce());
        if (rrProb < 1.0) {
            if (rand(state.RngState) > rrProb) {
                if ((payload.Flags & uint(BounceType.BOUNCE_TRANSMIT)) != 0u && hiddenEnvironmentOutputsTransparent())
                    clearPathVisibility(state);
                pathStates[sampleIndex] = state;
                return;
            }
            state.Throughput /= rrProb;
        }

        state.RayOrigin    = offsetRayOrigin(surface.worldPosition, geometricFacingNormal, payload.NextDirection);
        state.RayDirection = payload.NextDirection;
        state.Depth++;
        enqueueContinuationRay(sampleIndex, state);
    }

    float3 filterRadiance(PathState state, float3 referenceRadiance)
    {
        return suppressFireflies(
            state.Radiance, referenceRadiance,
            int(getDiffuseCount(state) + getSpecularCount(state) + getTransmissionCount(state)),
            int(getDiffuseCount(state)),
            int(getSpecularCount(state)),
            int(getTransmissionCount(state)));
    }
};

// ── Integrator dispatch macros ────────────────────────────────────────────────

#if defined(INTEGRATOR_AO)
#define ACTIVE_INTEGRATOR AoIntegrator
#else
#define ACTIVE_INTEGRATOR PathTracingIntegrator
#endif

uint    integratorSampleCount()                                                                  { ACTIVE_INTEGRATOR i; return i.sampleCount(); }
void    integratorShadeMiss(float3 dir, inout PathState s)                                       { ACTIVE_INTEGRATOR i; i.shadeMiss(dir, s); }
void    integratorShadeSurface(SurfaceData sf, float3 dir, uint idx, inout PathState s, inout Payload p) { ACTIVE_INTEGRATOR i; i.shadeSurface(sf, dir, idx, s, p); }
float3  integratorFilterRadiance(PathState s, float3 ref)                                        { ACTIVE_INTEGRATOR i; return i.filterRadiance(s, ref); }

#endif // RAYTRACING_INTEGRATORS_SLANG
