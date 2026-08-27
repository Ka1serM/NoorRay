#pragma once

// A 1:1 port of Materials/Shading/Dielectric.h.

static const float DielectricDenominatorEpsilon = 1.0e-5;

// Exact unpolarized Fresnel reflectance at a dielectric interface.
float dielectricFresnel(float incidentCosine, float incidentIor, float transmittedIor)
{
    const float cosine = min(abs(incidentCosine), 1.0);
    const float relativeIor = transmittedIor / max(incidentIor, DielectricDenominatorEpsilon);
    float g = relativeIor * relativeIor - 1.0 + cosine * cosine;
    if (g <= 0.0)
        return 1.0;

    g = sqrt(g);
    const float gPlusCosine = g + cosine;
    const float a = (g - cosine) / max(gPlusCosine, DielectricDenominatorEpsilon);
    const float b = (cosine * gPlusCosine - 1.0)
        / max(abs(cosine * (g - cosine) + 1.0), DielectricDenominatorEpsilon);
    return clamp(0.5 * a * a * (1.0 + b * b), 0.0, 1.0);
}

// Exact conductor Fresnel, kept beside the dielectric equations so measured
// spectral n/k data can be introduced without another helper.
float conductorFresnel(float incidentCosine, float eta, float extinction)
{
    const float cosine = min(abs(incidentCosine), 1.0);
    const float cosineSquared = cosine * cosine;
    const float sineSquared = max(1.0 - cosineSquared, 0.0);
    const float etaSquared = eta * eta;
    const float extinctionSquared = extinction * extinction;

    const float t0 = etaSquared - extinctionSquared - sineSquared;
    const float a2PlusB2 = sqrt(max(t0 * t0 + 4.0 * etaSquared * extinctionSquared, 0.0));
    const float a = sqrt(max(0.5 * (a2PlusB2 + t0), 0.0));
    const float t1 = a2PlusB2 + cosineSquared;
    const float t2 = 2.0 * cosine * a;
    const float rs = (t1 - t2) / max(t1 + t2, DielectricDenominatorEpsilon);
    const float t3 = cosineSquared * a2PlusB2 + sineSquared * sineSquared;
    const float t4 = t2 * sineSquared;
    const float rp = rs * (t3 - t4) / max(t3 + t4, DielectricDenominatorEpsilon);
    return clamp(0.5 * (rs + rp), 0.0, 1.0);
}

float iorFromNormalReflectance(float reflectance)
{
    const float root = sqrt(clamp(reflectance, 0.0, 0.999999));
    return (1.0 + root) / max(1.0 - root, DielectricDenominatorEpsilon);
}

// Colored F0 conductor approximation used by the metallic workflow.
float fresnelFromNormalReflectance(float cosine, float reflectance)
{
    return dielectricFresnel(cosine, 1.0, iorFromNormalReflectance(reflectance));
}

// Cosine-weighted hemispherical average of exact dielectric Fresnel.
float averageFresnel(float relativeIor)
{
    const float eta = max(relativeIor, DielectricDenominatorEpsilon);
    const float average = eta < 1.0
        ? 0.997118 + eta * (0.1014 - eta * (0.965241 + eta * 0.130607))
        : (eta - 1.0) / (4.08567 + 1.00071 * eta);
    return clamp(average, 0.0, 1.0);
}
