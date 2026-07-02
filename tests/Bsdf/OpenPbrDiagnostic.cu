#include <cstdio>
#include "OpenPbr/OpenPbrInputs.h"
#include "OpenPbr/OpenPbrLuts.h"
#include "Samplers/RandomSampler.h"

__global__ void diagnosticKernel()
{
    Material material{};
    material.albedo = glm::vec3(0.8f, 0.8f, 0.8f);
    material.specular = 0.5f;
    material.metallic = 0.0f;
    material.roughness = 0.5f;
    material.transmission = 0.0f;

    glm::vec2 uv(0.5f, 0.5f);
    glm::vec3 shadingNormal(0.0f, 0.0f, 1.0f);
    glm::vec3 tangent(1.0f, 0.0f, 0.0f);
    glm::vec3 viewDirection(0.0f, 0.0f, 1.0f);

    const float ior = sellmeierIor(material.sellmeier, 545.9f);
    printf("ior=%f\n", ior);

    const float weighted_specular_ior = openpbr_apply_specular_weight_to_ior(ior, material.specular);
    printf("weighted_specular_ior=%f\n", weighted_specular_ior);

    const float alpha = 0.5f * 0.5f; // roughness^2
    const float cos_in = 1.0f;

    const float num = openpbr_look_up_opaque_dielectric_energy_complement(weighted_specular_ior, alpha, cos_in);
    const float den = openpbr_look_up_opaque_dielectric_average_energy_complement(weighted_specular_ior, alpha);
    printf("energy_complement num=%f den=%f clamp(den)=%f\n",
        num, den, openpbr_clamp_average_energy_complement_above_zero(den));

    // Raw table indices used
    printf("exact_index_ior=%f exact_index_alpha=%f exact_index_costheta=%f\n",
        openpbr_ior_to_exact_index(weighted_specular_ior),
        openpbr_alpha_to_exact_index(alpha),
        openpbr_cos_theta_to_exact_index(cos_in));
}

int main()
{
    uploadOpenPbrLuts();
    diagnosticKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    destroyOpenPbrLuts();
    return 0;
}
