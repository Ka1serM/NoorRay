import NoorRHI;

//! n = a + b / lambda^2 (lambda in micrometres), fitted so that n(587.6 nm) = ior and 1 / Abbe = inverseAbbe; matches Spectral/Sellmeier.h Cauchy::fromAbbe.
float cauchyIor(float ior, float inverseAbbe, float lambdaNm) {
    const float b = (ior - 1.0) * inverseAbbe / (1.0 / (0.4861 * 0.4861) - 1.0 / (0.6563 * 0.6563));
    const float a = ior - b / (0.5876 * 0.5876);
    const float l = lambdaNm * 1e-3;
    return a + b / (l * l);
}

float4 cauchyIor(float ior, float inverseAbbe, float4 lambdaNm) {
    return float4(cauchyIor(ior, inverseAbbe, lambdaNm.x), cauchyIor(ior, inverseAbbe, lambdaNm.y), cauchyIor(ior, inverseAbbe, lambdaNm.z),
                  cauchyIor(ior, inverseAbbe, lambdaNm.w));
}
