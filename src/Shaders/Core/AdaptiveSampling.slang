// Adaptive sampling functions
#ifndef ADAPTIVE_SAMPLING_SLANG
#define ADAPTIVE_SAMPLING_SLANG

#include "../Shared.h"
#include "../Bindings.slang"
#include "../Core/Core.slang"
#include "Accumulation.slang"
#include "RenderPolicy.slang"

struct AdaptiveSamplingGateContext {
    bool adaptiveSamplingEnabled;
    bool wasConverged;
};

struct AdaptiveSamplingFrameContext {
    float exposureScale;
    bool adaptiveSamplingEnabled;
};

struct AdaptiveSamplingContext {
    float4 prevColorData;
    float4 prevAdaptiveData;
    float exposureScale;
    bool adaptiveSamplingEnabled;
    int adaptiveMinSamples;
    float adaptiveTargetError;
    float previousTotalSamples;
};

struct AdaptiveSamplingFrameState {
    float metricMean;
    float metricM2;
    int samplesTaken;
};

AdaptiveSamplingGateContext loadAdaptiveSamplingGateContext(int2 pixelCoord)
{
    AdaptiveSamplingGateContext context;
    context.adaptiveSamplingEnabled =
        (sceneSettings.renderSettings.adaptiveSamplingEnabled != 0) &&
        (pushConstants.isMoving == 0);
    context.wasConverged = false;
    if (!context.adaptiveSamplingEnabled || pushConstants.frame <= 0)
        return context;

    float4 previousAdaptiveData = outputAdaptiveState[pixelCoord];
    float previousTotalSamples = previousAdaptiveData.z;
    float adaptiveMinSamples = float(max(1, sceneSettings.renderSettings.adaptiveMinSamples));
    context.wasConverged =
        previousTotalSamples >= adaptiveMinSamples &&
        previousAdaptiveData.w > 0.5;
    return context;
}

AdaptiveSamplingFrameContext loadAdaptiveSamplingFrameContext()
{
    AdaptiveSamplingFrameContext context;
    context.exposureScale = exp2(sceneSettings.renderSettings.exposure);
    context.adaptiveSamplingEnabled =
        (sceneSettings.renderSettings.adaptiveSamplingEnabled != 0) &&
        (pushConstants.isMoving == 0);
    return context;
}

AdaptiveSamplingContext loadAdaptiveSamplingContext(int2 pixelCoord)
{
    AdaptiveSamplingContext context;
    bool hasHistory = (pushConstants.frame > 0) && (pushConstants.isMoving == 0);
    context.prevColorData = hasHistory ? outputColor[pixelCoord] : float4(0.0);
    context.prevAdaptiveData = hasHistory ? outputAdaptiveState[pixelCoord] : float4(0.0);
    context.exposureScale = exp2(sceneSettings.renderSettings.exposure);
    context.adaptiveSamplingEnabled =
        (sceneSettings.renderSettings.adaptiveSamplingEnabled != 0) &&
        (pushConstants.isMoving == 0);
    context.adaptiveMinSamples = max(1, sceneSettings.renderSettings.adaptiveMinSamples);
    context.adaptiveTargetError = max(sceneSettings.renderSettings.adaptiveTargetError, 0.0);
    context.previousTotalSamples = hasHistory ? context.prevAdaptiveData.z : 0.0;
    return context;
}

AdaptiveSamplingFrameState createAdaptiveSamplingFrameState()
{
    AdaptiveSamplingFrameState state;
    state.metricMean = 0.0;
    state.metricM2 = 0.0;
    state.samplesTaken = 0;
    return state;
}

bool shouldSkipAdaptiveSampling(AdaptiveSamplingGateContext context)
{
    return context.adaptiveSamplingEnabled && context.wasConverged;
}

void updateAdaptiveSamplingState(
    in AdaptiveSamplingFrameContext context,
    in float3 sampleColor,
    inout AdaptiveSamplingFrameState state)
{
    state.samplesTaken++;
    if (!context.adaptiveSamplingEnabled)
        return;

    float sampleLuminance = max(luminance(sampleColor), EPSILON);
    float sampleMetric = log2(max(1.0 + sampleLuminance * context.exposureScale, EPSILON));

    float sampleCountF = float(state.samplesTaken);
    float metricDelta = sampleMetric - state.metricMean;
    state.metricMean += metricDelta / sampleCountF;
    state.metricM2 += metricDelta * (sampleMetric - state.metricMean);
}

float combineAdaptiveMean(float historyMean, float historySamples, float frameMean, int frameSamples)
{
    if (frameSamples <= 0)
        return historyMean;
    if (historySamples <= 0.0)
        return frameMean;

    float frameSamplesF = float(frameSamples);
    float totalSamples = historySamples + frameSamplesF;
    return historyMean + (frameMean - historyMean) * (frameSamplesF / totalSamples);
}

float combineAdaptiveM2(float historyMean, float historyM2, float historySamples, float frameMean, float frameM2, int frameSamples)
{
    if (frameSamples <= 0)
        return historyM2;
    if (historySamples <= 0.0)
        return frameM2;

    float frameSamplesF = float(frameSamples);
    float totalSamples = historySamples + frameSamplesF;
    float meanDelta = frameMean - historyMean;
    return historyM2 + frameM2 + meanDelta * meanDelta * historySamples * frameSamplesF / totalSamples;
}

float computeAdaptiveRelativeError(float metricMean, float metricM2, float totalSamples)
{
    if (totalSamples <= 0.0)
        return 1e30;

    float variance = totalSamples > 1.0 ? metricM2 / max(totalSamples - 1.0, 1.0) : 0.0;
    float standardError = sqrt(max(variance, 0.0) / max(totalSamples, 1.0));
    return standardError / max(abs(metricMean), EPSILON);
}

void finalizeAdaptiveSampling(
    in AdaptiveSamplingContext context,
    in AdaptiveSamplingFrameState frameState,
    in float3 accumulatedColor,
    in float newAlpha,
    out float4 finalColorData,
    out float4 finalAdaptiveData)
{
    float newSamplesF = float(max(frameState.samplesTaken, 1));
    float3 newColor = accumulatedColor / newSamplesF;

    float historySamples = context.previousTotalSamples;
    float totalSamples = historySamples + newSamplesF;

    float3 prevColor = context.prevColorData.rgb;
    float prevAlpha = context.prevColorData.a;
    float3 newColorWithExposure = newColor * context.exposureScale;
    float3 finalColor = accumulateWeightedColor(
        prevColor,
        historySamples,
        newColorWithExposure,
        newSamplesF);
    float finalAlpha = accumulateWeightedScalar(
        prevAlpha,
        historySamples,
        newAlpha,
        newSamplesF);

    // finalColor is associated/premultiplied by finalAlpha. The tonemapper
    // converts to straight alpha for viewport display and image export.
    if (!context.adaptiveSamplingEnabled) {
        finalColorData = float4(finalColor, finalAlpha);
        finalAdaptiveData = float4(0.0, 0.0, totalSamples, 0.0);
        return;
    }

    float combinedMetricMean = combineAdaptiveMean(context.prevAdaptiveData.x, historySamples, frameState.metricMean, frameState.samplesTaken);
    float combinedMetricM2 = combineAdaptiveM2(context.prevAdaptiveData.x, context.prevAdaptiveData.y, historySamples, frameState.metricMean, frameState.metricM2, frameState.samplesTaken);
    float combinedRelativeError = computeAdaptiveRelativeError(combinedMetricMean, combinedMetricM2, totalSamples);
    float converged = context.adaptiveSamplingEnabled &&
                      totalSamples >= float(context.adaptiveMinSamples) &&
                      combinedRelativeError <= context.adaptiveTargetError ? 1.0 : 0.0;
    finalColorData = float4(finalColor, finalAlpha);
    finalAdaptiveData = float4(combinedMetricMean, combinedMetricM2, totalSamples, converged);
}

#endif // ADAPTIVE_SAMPLING_SLANG
