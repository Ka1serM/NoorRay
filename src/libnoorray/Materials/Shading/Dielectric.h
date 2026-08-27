#pragma once

#include <cmath>

#include "Backend/Host/Platform.h"

namespace nr::shading::dielectric
{

inline constexpr float DenominatorEpsilon = 1.0e-5f;

// Exact unpolarized Fresnel reflectance at a dielectric interface.
NR_CPU_GPU inline float fresnel(
    const float incidentCosine,
    const float incidentIor,
    const float transmittedIor)
{
    const float cosine = fminf(fabsf(incidentCosine), 1.0f);
    const float relativeIor = transmittedIor
        / fmaxf(incidentIor, DenominatorEpsilon);
    float g = relativeIor * relativeIor - 1.0f + cosine * cosine;
    if (g <= 0.0f)
        return 1.0f;

    g = sqrtf(g);
    const float gPlusCosine = g + cosine;
    const float a = (g - cosine)
        / fmaxf(gPlusCosine, DenominatorEpsilon);
    const float b = (cosine * gPlusCosine - 1.0f)
        / fmaxf(fabsf(cosine * (g - cosine) + 1.0f),
            DenominatorEpsilon);
    return fminf(fmaxf(0.5f * a * a * (1.0f + b * b), 0.0f), 1.0f);
}

// Exact conductor Fresnel is kept beside the dielectric interface equations
// so measured spectral n/k data can be introduced without another helper.
NR_CPU_GPU inline float conductorFresnel(
    const float incidentCosine, const float eta, const float extinction)
{
    const float cosine = fminf(fabsf(incidentCosine), 1.0f);
    const float cosineSquared = cosine * cosine;
    const float sineSquared = fmaxf(1.0f - cosineSquared, 0.0f);
    const float etaSquared = eta * eta;
    const float extinctionSquared = extinction * extinction;

    const float t0 = etaSquared - extinctionSquared - sineSquared;
    const float a2PlusB2 = sqrtf(fmaxf(
        t0 * t0 + 4.0f * etaSquared * extinctionSquared, 0.0f));
    const float a = sqrtf(fmaxf(0.5f * (a2PlusB2 + t0), 0.0f));
    const float t1 = a2PlusB2 + cosineSquared;
    const float t2 = 2.0f * cosine * a;
    const float rs = (t1 - t2) / fmaxf(t1 + t2, DenominatorEpsilon);
    const float t3 = cosineSquared * a2PlusB2 + sineSquared * sineSquared;
    const float t4 = t2 * sineSquared;
    const float rp = rs * (t3 - t4)
        / fmaxf(t3 + t4, DenominatorEpsilon);
    return fminf(fmaxf(0.5f * (rs + rp), 0.0f), 1.0f);
}

NR_CPU_GPU inline float iorFromNormalReflectance(const float reflectance)
{
    const float root = sqrtf(fminf(fmaxf(reflectance, 0.0f), 0.999999f));
    return (1.0f + root) / fmaxf(1.0f - root, DenominatorEpsilon);
}

// Colored F0 conductor approximation used by the metallic workflow.
NR_CPU_GPU inline float fresnelFromNormalReflectance(
    const float cosine, const float reflectance)
{
    return fresnel(cosine, 1.0f, iorFromNormalReflectance(reflectance));
}

// Cosine-weighted hemispherical average of exact dielectric Fresnel.
NR_CPU_GPU inline float averageFresnel(const float relativeIor)
{
    const float eta = fmaxf(relativeIor, DenominatorEpsilon);
    const float average = eta < 1.0f
        ? 0.997118f + eta
            * (0.1014f - eta * (0.965241f + eta * 0.130607f))
        : (eta - 1.0f) / (4.08567f + 1.00071f * eta);
    return fminf(fmaxf(average, 0.0f), 1.0f);
}

}
