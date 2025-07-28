#pragma once

#include <glm/glm.hpp>
#include <cstdint>

#define PI 3.14159265359f
#define EPSILON 1e-6f

#define BOUNCE_TYPE_DIFFUSE 0
#define BOUNCE_TYPE_SPECULAR 1
#define BOUNCE_TYPE_TRANSMISSION 2

namespace ShadersCpu {

    // Interpolation
    glm::vec3 interpolateBarycentric(const glm::vec3& bary, const glm::vec3& v0,  const glm::vec3& v1, const glm::vec3& v2);
    glm::vec2 interpolateBarycentric(const glm::vec3& bary, const glm::vec2& v0, const glm::vec2& v1, const glm::vec2& v2);

    // Random Number Generation
    float rand(uint32_t& state);
    glm::uvec2 pcg2d(glm::uvec2 v);
    glm::vec2 roundBokeh(float u1, float u2, float bias);

    // Coordinate System
    void buildCoordinateSystem(const glm::vec3& N, glm::vec3& T, glm::vec3& B);

    // Diffuse BSDF
    glm::vec3 sampleDiffuse(const glm::vec3& N, uint32_t& rngState);
    float pdfDiffuse(const glm::vec3& N, const glm::vec3& L);
    glm::vec3 evaluateDiffuseBRDF(const glm::vec3& albedo, float metallic);

    // Specular (GGX) BSDF
    glm::vec3 sampleGGX(float roughness, const glm::vec3& N, uint32_t& rngState);
    float distributionGGX(const glm::vec3& N, const glm::vec3& H, float roughness);
    glm::vec3 sampleSpecular(const glm::vec3& viewDir, const glm::vec3& N, float roughness, uint32_t& rngState);
    float pdfSpecular(const glm::vec3& viewDir, const glm::vec3& N, float roughness, const glm::vec3& L);
    glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0);
    float geometrySchlickGGX(float NdotV, float roughness);
    float geometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float roughness);
    glm::vec3 evaluateSpecularBRDF(const glm::vec3& viewDir, const glm::vec3& N, const glm::vec3& albedo, float metallic, float roughness, const glm::vec3& L);

} // namespace ShaderHelpers