#ifndef ACCUMULATION_SLANG
#define ACCUMULATION_SLANG

float accumulateWeightedScalar(float historyValue, float historyWeight, float newValue, float newWeight)
{
    float invWeight = 1.0 / max(historyWeight + newWeight, EPSILON);
    return (historyValue * historyWeight + newValue * newWeight) * invWeight;
}

float3 accumulateWeightedColor(float3 historyColor, float historyWeight, float3 newColor, float newWeight)
{
    float invWeight = 1.0 / max(historyWeight + newWeight, EPSILON);
    return (historyColor * historyWeight + newColor * newWeight) * invWeight;
}

float loadHistorySamples(int2 pixelCoord)
{
    if (pushConstants.frame <= 0 || pushConstants.isMoving != 0)
        return 0.0;

    return max(0.0, outputAdaptiveState[pixelCoord].z);
}

float loadExposureScale()
{
    return exp2(sceneSettings.renderSettings.exposure);
}

float3 loadTemporalReferenceColor(int2 pixelCoord, float exposureScale)
{
    if (pushConstants.frame <= 0 || pushConstants.isMoving != 0 || pushConstants.pixelSizePercent > 100)
        return float3(0.0);

    return outputColor[pixelCoord].rgb / max(exposureScale, EPSILON);
}

void storeAccumulatedFrame(
    int2 pixelCoord,
    float historySamples,
    float3 newColorWithExposure,
    float newAlpha,
    float newWeight)
{
    float3 finalColor = accumulateWeightedColor(
        outputColor[pixelCoord].rgb,
        historySamples,
        newColorWithExposure,
        newWeight);
    float finalAlpha = accumulateWeightedScalar(
        outputColor[pixelCoord].a,
        historySamples,
        newAlpha,
        newWeight);

    outputColor[pixelCoord] = float4(finalColor, finalAlpha);
    outputAdaptiveState[pixelCoord] = float4(0.0, 0.0, historySamples + newWeight, 0.0);
}

#endif // ACCUMULATION_SLANG
