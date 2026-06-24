// Render policy functions
#ifndef RENDER_POLICY_SLANG
#define RENDER_POLICY_SLANG

#include "../Shared.h"
#include "../Bindings.slang"
#include "../Core/Core.slang"

bool isInteractiveFrame()
{
    return pushConstants.isMoving != 0;
}

int effectivePrimarySampleCount()
{
    return max(sceneSettings.renderSettings.samples, 1);
}

bool useShaderPixelSize()
{
    return pushConstants.pixelSizePercent > 100;
}

int ceilDivInt(int numerator, int denominator)
{
    return (numerator + denominator - 1) / max(denominator, 1);
}

int partitionStart(int partitionIndex, int fullSize, int partitionCount)
{
    return ceilDivInt(partitionIndex * fullSize, partitionCount);
}

int partitionEndExclusive(int partitionIndex, int fullSize, int partitionCount)
{
    return ceilDivInt((partitionIndex + 1) * fullSize, partitionCount);
}

int2 effectiveShadingScreenSize(int2 fullScreenSize)
{
    if (!useShaderPixelSize())
        return fullScreenSize;

    int percent = max(pushConstants.pixelSizePercent, 100);
    int width = max(1, ceilDivInt(fullScreenSize.x * 100, percent));
    int height = max(1, ceilDivInt(fullScreenSize.y * 100, percent));
    return int2(width, height);
}

int2 mapOutputPixelToShadingPixel(int2 outputPixel, int2 fullScreenSize, int2 shadingScreenSize)
{
    if (all(shadingScreenSize == fullScreenSize))
        return outputPixel;

    int x = (outputPixel.x * shadingScreenSize.x) / max(fullScreenSize.x, 1);
    int y = (outputPixel.y * shadingScreenSize.y) / max(fullScreenSize.y, 1);
    return clamp(int2(x, y), int2(0), shadingScreenSize - 1);
}

int2 shadingPixelStart(int2 shadingPixel, int2 fullScreenSize, int2 shadingScreenSize)
{
    int x = partitionStart(shadingPixel.x, fullScreenSize.x, shadingScreenSize.x);
    int y = partitionStart(shadingPixel.y, fullScreenSize.y, shadingScreenSize.y);
    return clamp(int2(x, y), int2(0), fullScreenSize - 1);
}

int2 shadingPixelEndExclusive(int2 shadingPixel, int2 fullScreenSize, int2 shadingScreenSize)
{
    int x = partitionEndExclusive(shadingPixel.x, fullScreenSize.x, shadingScreenSize.x);
    int y = partitionEndExclusive(shadingPixel.y, fullScreenSize.y, shadingScreenSize.y);
    return clamp(int2(x, y), int2(1), fullScreenSize);
}

bool isShadingLeaderPixel(int2 outputPixel, int2 fullScreenSize, int2 shadingScreenSize)
{
    int2 shadingPixel = mapOutputPixelToShadingPixel(outputPixel, fullScreenSize, shadingScreenSize);
    int2 cellStart = shadingPixelStart(shadingPixel, fullScreenSize, shadingScreenSize);
    return all(outputPixel == cellStart);
}

int effectiveAoSampleCount()
{
    return max(sceneSettings.renderSettings.samples, 1);
}

int effectiveMaxBounceCount()
{
    int configuredMax = max(
        sceneSettings.renderSettings.diffuseBounces,
        max(sceneSettings.renderSettings.specularBounces, sceneSettings.renderSettings.transmissionBounces));
    return configuredMax;
}

int effectiveRussianRouletteStartBounce()
{
    if (isInteractiveFrame())
        return 64;
    return max(1, sceneSettings.renderSettings.russianRouletteStartBounce);
}

bool useInteractivePathTracingPreview()
{
    return useShaderPixelSize();
}

#endif // RENDER_POLICY_SLANG
