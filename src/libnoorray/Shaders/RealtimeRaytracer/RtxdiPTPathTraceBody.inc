void RAB_PathTrace(inout PT_CONTEXT context,
    inout RTXDI_PathTracerRandomContext random,
    inout RAB_PathTracerUserData userData)
{
    while (context.GetBounceDepth() <= context.GetMaxPathBounce())
    {
        context.BeginPathState();
        const RAB_Surface previousSurface = context.GetIntersectionSurface();
        context.SetBrdfRaySample(realtimePTSampleBsdf(previousSurface,
            random.replayRandomSamplerState));
        if (!context.ValidContinuationRayBrdfOverPdf())
            break;
        context.MultiplyPathThroughput(context.GetContinuationRayBrdfOverPdf());
        // New paths end once they carry too little to matter; replays of
        // reused paths must follow them to the end.
        if (userData.pathType == RTXDI_PTPathTraceInvocationType_Initial
            && dot(context.GetPathThroughput(), context.GetPathThroughput())
                < RealtimePTMinimumThroughput * RealtimePTMinimumThroughput)
            break;
        if (context.ShouldRunRussianRoulette())
        {
            const float continuation = RealtimePTRussianRouletteContinuation;
            if (RTXDI_GetNextRandom(random.initialRandomSamplerState) > continuation)
                break;
            context.MultiplyPathThroughput(1.0 / continuation);
            context.RecordRussianRouletteProbability(continuation);
        }
        RayDesc ray;
        ray.Origin = offsetRayOrigin(previousSurface.worldPos, previousSurface.geoNormal,
            context.GetBrdfRaySample().outDirection);
        ray.Direction = context.GetBrdfRaySample().outDirection;
        ray.TMin = 1.0e-3;
        ray.TMax = 1.0e30;
        context.SetContinuationRay(ray);
        if (!context.AnalyzePathReconnectibilityBeforeTrace())
            break;
        // Paths pass fractional layers, which dim them like everywhere else
        // in the renderer; the layers' own light is left out.
        const TranslucentLayers layers = traceTranslucentLayers(ray,
            leaveSurface(previousSurface, ray.Direction,
                context.GetBrdfRaySample().properties.IsSpecular()),
            previousSurface.pathRoughness);
        RAB_RayPayload traced;
        traced.committedRayT = layers.surface.t;
        context.SetTraceResult(traced);
        context.MultiplyPathThroughput(float3(layers.transmittance));
        if (layers.surface.instance == RealtimeMissInstance)
        {
            context.RecordPathRadianceMiss(random.initialRandomSamplerState);
            // The ray leaving the primary surface carries the environment's
            // direct light there, which ReSTIR DI owns. Later rays share the
            // environment with light sampling.
            Frame frame = realtimeArgs()->frame;
            if (frame.environment != nullptr && context.GetBounceDepth() != 2u)
                context.RecordEnvironmentMapLightSample(environmentEscapedRadiance(
                    frame.environment[0], ray.Direction, previousSurface.pathRoughness)
                    * (1.0 - realtimeEnvironmentLightWeight(ray.Direction)),
                    previousSurface, random.initialRandomSamplerState);
            break;
        }
        // Unreal's path roughness: the roughest lobe scattered from so far,
        // diffuse counting as 1.
        const float lobeRoughness = context.GetBrdfRaySample().properties.IsSpecular()
            ? RAB_GetSurfaceRoughness(previousSurface) : 1.0;
        const RAB_Surface surface = realtimePTHitSurface(layers.surface, ray,
            max(previousSurface.pathRoughness, lobeRoughness));
        context.RecordPathIntersection(surface);
        if (context.IsPathTerminated())
            break;
        if (context.ShouldSampleEmissiveSurfaces())
            context.RecordEmissiveLightSample(layers.surface.emission,
                previousSurface, random.initialRandomSamplerState);
        if (context.ShouldSampleNee())
        {
            RTXDI_SampledLightData sampledLight;
            float3 radiance;
            float neePdf;
            float scatterPdf;
            RAB_LightSample lightSample;
            realtimePTSampleNEE(surface, random, sampledLight, radiance,
                neePdf, scatterPdf, lightSample);
            if (neePdf > 0.0)
                context.RecordNeeLightSample(sampledLight, radiance, neePdf,
                    scatterPdf, lightSample, random.initialRandomSamplerState);
        }
        context.IncreaseBounceDepth();
    }
}
