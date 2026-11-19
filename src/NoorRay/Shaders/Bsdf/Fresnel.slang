import NoorRHI;
import Common.Math;

//! Exact unpolarized reflectance for a cosine on the incident side and eta = n_transmitted / n_incident; one under total internal reflection.
float fresnelDielectric(float cosine, float eta) {
    const float c = min(abs(cosine), 1.0);
    float g = eta * eta - 1.0 + c * c;
    if (g <= 0.0)
        return 1.0;
    g = sqrt(g);
    const float a = (g - c) / (g + c);
    const float b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return saturate(0.5 * a * a * (1.0 + b * b));
}

float4 fresnelDielectric(float cosine, float4 eta) {
    return float4(fresnelDielectric(cosine, eta.x), fresnelDielectric(cosine, eta.y), fresnelDielectric(cosine, eta.z), fresnelDielectric(cosine, eta.w));
}

float f0FromEta(float eta) { return square((eta - 1.0) / (eta + 1.0)); }

float etaFromF0(float f0) {
    const float root = sqrt(clamp(f0, 0.0, 0.9999));
    return (1.0 + root) / (1.0 - root);
}

//! The cosine-weighted hemispherical average of fresnelDielectric (Kulla and Conty 2017 fit).
float averageFresnelDielectric(float eta) {
    if (eta < 1.0)
        return saturate(0.997118 + eta * (0.1014 - eta * (0.965241 + eta * 0.130607)));
    return saturate((eta - 1.0) / (4.08567 + 1.00071 * eta));
}

//! The F82-tint conductor of Blender's Principled BSDF (Kutz et al. 2021): Schlick from f0, pulled towards f82 = tint * Schlick(1/7) at cos = 1/7.
float fresnelF82(float cosine, float f0, float tint) {
    const float mu = saturate(cosine);
    const float schlick = f0 + (1.0 - f0) * pow(1.0 - mu, 5.0);
    const float barMu = 1.0 / 7.0;
    const float schlickBar = f0 + (1.0 - f0) * pow(1.0 - barMu, 5.0);
    const float a = schlickBar * (1.0 - tint) / (barMu * pow(1.0 - barMu, 6.0));
    return saturate(schlick - a * mu * pow(1.0 - mu, 6.0));
}

float averageFresnelF82(float f0, float tint) {
    const float barMu = 1.0 / 7.0;
    const float schlickBar = f0 + (1.0 - f0) * pow(1.0 - barMu, 5.0);
    const float a = schlickBar * (1.0 - tint) / (barMu * pow(1.0 - barMu, 6.0));
    return saturate(f0 + (1.0 - f0) / 21.0 - a / 126.0);
}

float airyComponent(float r12, float r23, float cosPhase) {
    const float numerator = r12 * r12 + r23 * r23 + 2.0 * r12 * r23 * cosPhase;
    const float denominator = 1.0 + r12 * r12 * r23 * r23 + 2.0 * r12 * r23 * cosPhase;
    return saturate(numerator / max(denominator, 1e-6));
}

//! Reflectance of a thin film (index filmEta, thickness in nm) over a substrate of real index substrateEta, seen from a medium of index one, at one wavelength.
float fresnelThinFilm(float cosine, float filmEta, float thicknessNm, float substrateEta, float lambdaNm) {
    const float cos1 = saturate(abs(cosine));
    const float sin1Squared = 1.0 - cos1 * cos1;
    const float sin2Squared = sin1Squared / (filmEta * filmEta);
    if (sin2Squared >= 1.0)
        return 1.0;
    const float cos2 = sqrt(1.0 - sin2Squared);
    const float sin3Squared = sin1Squared / (substrateEta * substrateEta);
    const float cos3 = sin3Squared >= 1.0 ? 0.0 : sqrt(1.0 - sin3Squared);
    const float r12s = (cos1 - filmEta * cos2) / (cos1 + filmEta * cos2);
    const float r12p = (filmEta * cos1 - cos2) / (filmEta * cos1 + cos2);
    const float r23s = (filmEta * cos2 - substrateEta * cos3) / (filmEta * cos2 + substrateEta * cos3);
    const float r23p = (substrateEta * cos2 - filmEta * cos3) / (substrateEta * cos2 + filmEta * cos3);
    const float cosPhase = cos(4.0 * pi * filmEta * thicknessNm * cos2 / lambdaNm);
    return 0.5 * (airyComponent(r12s, r23s, cosPhase) + airyComponent(r12p, r23p, cosPhase));
}
