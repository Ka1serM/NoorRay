#ifndef IBXDF_SLANG
#define IBXDF_SLANG

struct BxdfEval
{
    float3 f;
    float pdf;
};

interface IBxdf
{
    BxdfEval evalOpaque(
        float3 viewDir,
        float3 perturbedNormal,
        float3 geometricNormal,
        float3 lightDir,
        float3 albedo,
        float specular,
        float metallic,
        float roughness,
        float swSpecular);
};

#endif // IBXDF_SLANG
