#include "ShadersCpu.h"
#include <cmath>
#include <algorithm>

namespace ShadersCpu {

glm::vec3 interpolateBarycentric(const glm::vec3& bary, const glm::vec3& v0,  const glm::vec3& v1, const glm::vec3& v2) {
    return bary.x * v0 + bary.y * v1 + bary.z * v2;
}

glm::vec2 interpolateBarycentric(const glm::vec3& bary, const glm::vec2& v0, const glm::vec2& v1, const glm::vec2& v2) {
    return bary.x * v0 + bary.y * v1 + bary.z * v2;
}

float rand(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return float(state) / float(0xFFFFFFFFu);
}

glm::uvec2 pcg2d(glm::uvec2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v = v ^ (v >> 16u);
    return v;
}

glm::vec2 roundBokeh(float u1, float u2, float bias) {
    float r = std::sqrt(u1);
    float theta = 2.0f * PI * u2;
    return glm::vec2(r * std::cos(theta), r * std::sin(theta));
}

void buildCoordinateSystem(const glm::vec3& N, glm::vec3& T, glm::vec3& B) {
    if (std::abs(N.z) < 0.999f) {
        T = glm::normalize(glm::cross(N, glm::vec3(0.0f, 0.0f, 1.0f)));
    } else {
        T = glm::normalize(glm::cross(N, glm::vec3(0.0f, 1.0f, 0.0f)));
    }
    B = glm::cross(T, N);
}

glm::vec3 sampleDiffuse(const glm::vec3& N, uint32_t& rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    float r = std::sqrt(u1);
    float theta = 2.0f * PI * u2;
    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float z = std::sqrt(std::max(0.0f, 1.0f - u1));
    glm::vec3 T, B;
    buildCoordinateSystem(N, T, B);
    return glm::normalize(x * T + y * B + z * N);
}

float pdfDiffuse(const glm::vec3& N, const glm::vec3& L) {
    float NoL = std::max(glm::dot(N, L), 0.0f);
    return NoL / PI;
}

glm::vec3 evaluateDiffuseBRDF(const glm::vec3& albedo, float metallic) {
    return albedo / PI * (1.0f - metallic);
}

glm::vec3 sampleGGX(float roughness, const glm::vec3& N, uint32_t& rngState) {
    float u1 = rand(rngState);
    float u2 = rand(rngState);
    float a = roughness * roughness;
    float phi = 2.0f * PI * u1;
    float denom = 1.0f + (a * a - 1.0f) * u2;
    denom = std::max(denom, EPSILON);
    float cosTheta = std::sqrt(std::max(0.0f, (1.0f - u2) / denom));
    float sinTheta = std::sqrt(std::max(0.0f, 1.0f - cosTheta * cosTheta));
    glm::vec3 Ht = glm::vec3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
    glm::vec3 T, B;
    buildCoordinateSystem(N, T, B);
    return glm::normalize(Ht.x * T + Ht.y * B + Ht.z * N);
}

float distributionGGX(const glm::vec3& N, const glm::vec3& H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = std::max(glm::dot(N, H), 0.0f);
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = std::max(denom, EPSILON);
    denom = PI * denom * denom;
    return nom / denom;
}

glm::vec3 sampleSpecular(const glm::vec3& viewDir, const glm::vec3& N, float roughness, uint32_t& rngState) {
    glm::vec3 H = sampleGGX(roughness, N, rngState);
    if (glm::dot(H, N) < 0.0f) {
        H = -H;
    }
    return glm::reflect(-viewDir, H);
}

float pdfSpecular(const glm::vec3& viewDir, const glm::vec3& N, float roughness, const glm::vec3& L) {
    glm::vec3 H = glm::normalize(viewDir + L);
    float NoH = std::max(glm::dot(N, H), EPSILON);
    float VoH = std::max(glm::dot(viewDir, H), EPSILON);
    float D = distributionGGX(N, H, roughness);
    return std::max((D * NoH) / (4.0f * VoH), EPSILON);
}

glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0) {
    cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);
    return F0 + (1.0f - F0) * std::pow(1.0f - cosTheta, 5.0f);
}

float geometrySchlickGGX(float NdotV, float roughness) {
    NdotV = std::max(NdotV, EPSILON);
    float k = (roughness * roughness) / 2.0f;
    return NdotV / (NdotV * (1.0f - k) + k);
}

float geometrySmith(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, float roughness) {
    float NdotV = std::max(glm::dot(N, V), 0.0f);
    float NdotL = std::max(glm::dot(N, L), 0.0f);
    float ggxV = geometrySchlickGGX(NdotV, roughness);
    float ggxL = geometrySchlickGGX(NdotL, roughness);
    return ggxV * ggxL;
}

glm::vec3 evaluateSpecularBRDF(const glm::vec3& viewDir, const glm::vec3& N, const glm::vec3& albedo, float metallic, float roughness, const glm::vec3& L) {
    glm::vec3 H = glm::normalize(viewDir + L);
    float NoV = std::max(glm::dot(N, viewDir), 0.0f);
    float NoL = std::max(glm::dot(N, L), 0.0f);
    float VoH = std::max(glm::dot(viewDir, H), 0.0f);
    float D = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, viewDir, L, roughness);
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
    glm::vec3 F = fresnelSchlick(VoH, F0);
    glm::vec3 numerator = F * D * G;
    float denominator = std::max(4.0f * NoV * NoL, EPSILON);
    return numerator / denominator;
}

} // namespace ShaderHelpers