#pragma once

#include <compare>
#include <string>

#include "NoorRay.Python/Blender/Reader.h"
#include "NoorRay/Scene/Texture.h"

namespace NoorRay::Python {

enum class SkyType { Preetham, Hosek, SingleScattering, MultipleScattering };

//! The settings of a Sky Texture node; the sun direction is a Blender Z-up vector, the angles are radians.
struct SkySettings {
    SkyType type = SkyType::MultipleScattering;
    bool sunDisc = true;
    float sunSize = 0.009512f;
    float sunIntensity = 1.0f;
    float sunElevation = 0.2617994f;
    float sunRotation = 0.0f;
    float altitude = 100.0f;
    float airDensity = 1.0f;
    float aerosolDensity = 1.0f;
    float ozoneDensity = 1.0f;
    float sunDirection[3] = {0.0f, 0.0f, 1.0f};
    float turbidity = 2.2f;
    float groundAlbedo = 0.3f;

    auto operator<=>(const SkySettings&) const = default;
};

//! The settings of a ShaderNodeTexSky.
SkySettings readSkySettings(Reader& reader, const Object& node);

//! The sky in linear Rec.709 on an equirectangular table, top row first; the texels at the sun are supersampled, so the table keeps the sun disc energy.
Scene::Texture makeSkyTexture(const SkySettings& settings, std::string name);

}
