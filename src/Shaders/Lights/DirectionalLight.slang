#ifndef DIRECTIONAL_LIGHT_SLANG
#define DIRECTIONAL_LIGHT_SLANG

#include "../Bindings.slang"
#include "../Core/Core.slang"
#include "ILight.slang"

bool hasDirectionalLight(in EnvironmentSettings environmentData) {
    return environmentData.directionalIntensity > EPSILON &&
           dot(environmentData.directionalDirection, environmentData.directionalDirection) > EPSILON;
}

float3 getDirectionalLightDirection(in EnvironmentSettings environmentData) {
    float3 d = environmentData.directionalDirection;
    float l2 = dot(d, d);
    if (l2 <= EPSILON)
        return float3(0.0, 1.0, 0.0);
    return d * rsqrt(l2);
}

// Builds an orthonormal tangent frame around a unit normal (Duff et al. 2017).
void buildOrthonormalBasis(in float3 n, out float3 tangent, out float3 bitangent) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    tangent = float3(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    bitangent = float3(b, s + n.y * n.y * a, -n.y);
}

// Samples a direction uniformly within a cone of the given half-angle (degrees)
// centered on the sun direction. A zero angle returns the exact direction,
// while larger angles spread samples wider and soften the shadow penumbra.
float3 sampleDirectionalLightDirection(in EnvironmentSettings environmentData, float2 u) {
    float3 baseDir = getDirectionalLightDirection(environmentData);
    float softAngle = environmentData.directionalSoftAngle;
    if (softAngle <= EPSILON)
        return baseDir;

    float cosThetaMax = cos(radians(softAngle));
    float cosTheta = 1.0 - u.x * (1.0 - cosThetaMax);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * PI * u.y;

    float3 tangent, bitangent;
    buildOrthonormalBasis(baseDir, tangent, bitangent);
    return normalize(
        tangent * (cos(phi) * sinTheta) +
        bitangent * (sin(phi) * sinTheta) +
        baseDir * cosTheta);
}

struct DirectionalLight : ILight
{
    LightLiSample sampleLi(float3 referencePoint, float2 u)
    {
        referencePoint = referencePoint;

        LightLiSample sample;
        sample.wi = sampleDirectionalLightDirection(sceneSettings.environment, u);
        sample.Li = float3(sceneSettings.environment.directionalIntensity);
        sample.pdf = hasDirectionalLight(sceneSettings.environment) ? 1.0 : 0.0;
        return sample;
    }

    float pdfLi(float3 referencePoint, float3 wi)
    {
        referencePoint = referencePoint;
        wi = wi;
        return 0.0;
    }

    LightLeSample sampleLe(float2 u1, float2 u2)
    {
        u1 = u1;
        u2 = u2;

        LightLeSample sample;
        sample.direction = getDirectionalLightDirection(sceneSettings.environment);
        sample.radiance = float3(sceneSettings.environment.directionalIntensity);
        sample.pdfPos = hasDirectionalLight(sceneSettings.environment) ? 1.0 : 0.0;
        sample.pdfDir = hasDirectionalLight(sceneSettings.environment) ? 1.0 : 0.0;
        return sample;
    }
};

#endif // DIRECTIONAL_LIGHT_SLANG
