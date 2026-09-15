#pragma once

#include <glm/vec3.hpp>

#include "Optics/Sellmeier.h"

// Compact host-side material parameters consumed by the SVM material builder.
// This is importer input, not shader state and is never copied to the device.
struct SvmMaterial
{
    glm::vec3 albedo{1.0f};
    int albedoIndex{-1};
    float specular{1.0f};
    float metallic{};
    float roughness{};
    SellmeierCoefficients sellmeier{};
    int specularIndex{-1};
    int metallicIndex{-1};
    int roughnessIndex{-1};
    int normalIndex{-1};
    glm::vec3 transmissionColor{1.0f};
    float transmission{};
    glm::vec3 emission{1.0f};
    float emissionStrength{};
    int emissionIndex{-1};
    int transmissionIndex{-1};
    int opacityIndex{-1};
    float opacity{1.0f};
};
