#pragma once

#include <expected>
#include <string>
#include <string_view>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Scene/Texture.h"

namespace NoorRay::Materials {

//! A photometric profile as Cycles' IESFile keeps it: radians, vertical angle 0 at the nadir; intensity has one run of vertical values per horizontal angle.
struct IesProfile {
    std::vector<float> horizontal;
    std::vector<float> vertical;
    std::vector<float> intensity;

    //! Cycles' kernel_ies_interp: cubic along both angles, and zero outside the angles of the file.
    float evaluate(float horizontalAngle, float verticalAngle) const;
};

[[nodiscard]] std::expected<IesProfile, Core::Error> parseIes(std::string_view text);

//! The profile on a grid of angles, top row first: u is the horizontal angle over 2 pi, v the vertical angle over pi.
Scene::Texture makeIesTexture(const IesProfile& profile, std::string name);

}
