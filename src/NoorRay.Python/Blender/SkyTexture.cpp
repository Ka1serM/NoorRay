#include "NoorRay.Python/Blender/SkyTexture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "NoorRay.Python/Blender/Parallel.h"
#include "sky_hosek.h"
#include "sky_nishita.h"

namespace NoorRay::Python {

namespace {

constexpr float pi = std::numbers::pi_v<float>;
constexpr float halfPi = pi / 2.0f;
constexpr float twoPi = 2.0f * pi;
constexpr std::uint32_t tableWidth = 2048;
constexpr std::uint32_t tableHeight = 1024;
constexpr std::uint32_t lutWidth = 512;
constexpr std::uint32_t lutHeight = 256;
constexpr std::uint32_t sunSamples = 16;

using Vector = std::array<float, 3>;

float dot(const Vector& a, const Vector& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
float length(const Vector& a) {
    return std::sqrt(dot(a, a));
}

//! Cycles' precise_angle between two unit vectors.
float angleBetween(const Vector& a, const Vector& b) {
    const Vector difference{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    const Vector sum{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    return 2.0f * std::atan2(length(difference), length(sum));
}

//! The XYZ to linear Rec.709 matrix of Blender's colour configuration, clamped at zero as Cycles' xyz_to_rgb_clamped.
Vector xyzToRgb(const Vector& xyz) {
    const float r = 3.2404373564920070f * xyz[0] - 1.5371305409000549f * xyz[1] - 0.4985288240756933f * xyz[2];
    const float g = -0.9692674982687597f * xyz[0] + 1.8760136862977035f * xyz[1] + 0.0415560804587997f * xyz[2];
    const float b = 0.0556434463874256f * xyz[0] - 0.2040259700872272f * xyz[1] + 1.0572254813610864f * xyz[2];
    return {std::max(r, 0.0f), std::max(g, 0.0f), std::max(b, 0.0f)};
}

//! Cycles' SunSky: the sun angles and the model coefficients of Preetham and Hosek-Wilkie.
struct Analytic {
    float theta = 0.0f;
    float phi = 0.0f;
    std::array<float, 3> radiance{};
    std::array<std::array<float, 9>, 3> config{};
};

float perez(const std::array<float, 9>& lam, const float theta, const float gamma) {
    return (1.0f + lam[0] * std::exp(lam[1] / std::cos(theta))) * (1.0f + lam[2] * std::exp(lam[3] * gamma) + lam[4] * std::cos(gamma) * std::cos(gamma));
}

float hosekInternal(const std::array<float, 9>& c, const float theta, const float gamma) {
    const float ctheta = std::cos(theta);
    const float cgamma = std::cos(gamma);
    const float rayM = cgamma * cgamma;
    const float mieM = (1.0f + rayM) / std::pow(1.0f + c[8] * c[8] - 2.0f * c[8] * cgamma, 1.5f);
    return (1.0f + c[0] * std::exp(c[1] / (ctheta + 0.01f))) * (c[2] + c[3] * std::exp(c[4] * gamma) + c[5] * rayM + c[6] * mieM + c[7] * std::sqrt(ctheta));
}

Vector normalized(const float (&direction)[3]) {
    const Vector v{direction[0], direction[1], direction[2]};
    const float size = length(v);
    if (size == 0.0f)
        return {0.0f, 0.0f, 1.0f};
    return {v[0] / size, v[1] / size, v[2] / size};
}

//! Cycles' sky_texture_precompute_preetham.
Analytic precomputePreetham(const SkySettings& settings) {
    const Vector sun = normalized(settings.sunDirection);
    Analytic a;
    a.theta = std::acos(std::clamp(sun[2], -1.0f, 1.0f));
    a.phi = std::atan2(sun[0], sun[1]);
    const float theta = a.theta;
    const float theta2 = theta * theta;
    const float theta3 = theta2 * theta;
    const float t = settings.turbidity;
    const float t2 = t * t;
    const float chi = (4.0f / 9.0f - t / 120.0f) * (pi - 2.0f * theta);
    a.radiance[0] = ((4.0453f * t - 4.9710f) * std::tan(chi) - 0.2155f * t + 2.4192f) * 0.06f;
    a.radiance[1] = (0.00166f * theta3 - 0.00375f * theta2 + 0.00209f * theta) * t2 + (-0.02903f * theta3 + 0.06377f * theta2 - 0.03202f * theta + 0.00394f) * t
                    + (0.11693f * theta3 - 0.21196f * theta2 + 0.06052f * theta + 0.25886f);
    a.radiance[2] = (0.00275f * theta3 - 0.00610f * theta2 + 0.00317f * theta) * t2 + (-0.04214f * theta3 + 0.08970f * theta2 - 0.04153f * theta + 0.00516f) * t
                    + (0.15346f * theta3 - 0.26756f * theta2 + 0.06670f * theta + 0.26688f);
    a.config[0] = {0.1787f * t - 1.4630f, -0.3554f * t + 0.4275f, -0.0227f * t + 5.3251f, 0.1206f * t - 2.5771f, -0.0670f * t + 0.3703f};
    a.config[1] = {-0.0193f * t - 0.2592f, -0.0665f * t + 0.0008f, -0.0004f * t + 0.2125f, -0.0641f * t - 0.8989f, -0.0033f * t + 0.0452f};
    a.config[2] = {-0.0167f * t - 0.2608f, -0.0950f * t + 0.0092f, -0.0079f * t + 0.2102f, -0.0441f * t - 1.6537f, -0.0109f * t + 0.0529f};
    for (std::size_t i = 0; i < 3; ++i)
        a.radiance[i] /= perez(a.config[i], 0.0f, theta);
    return a;
}

//! Cycles' sky_texture_precompute_hosek.
Analytic precomputeHosek(const SkySettings& settings) {
    const Vector sun = normalized(settings.sunDirection);
    Analytic a;
    a.theta = std::clamp(std::acos(std::clamp(sun[2], -1.0f, 1.0f)), 0.0f, halfPi);
    a.phi = std::atan2(sun[0], sun[1]);
    SKY_ArHosekSkyModelState* state =
        SKY_arhosek_xyz_skymodelstate_alloc_init(std::clamp(settings.turbidity, 0.0f, 10.0f), settings.groundAlbedo, halfPi - a.theta);
    for (std::size_t c = 0; c < 3; ++c) {
        for (std::size_t i = 0; i < 9; ++i)
            a.config[c][i] = static_cast<float>(state->configs[c][i]);
        a.radiance[c] = static_cast<float>(state->radiances[c]);
    }
    SKY_arhosekskymodelstate_free(state);
    return a;
}

//! Cycles' sky_radiance_preetham and sky_radiance_hosek.
Vector analyticRadiance(const SkySettings& settings, const Analytic& a, const Vector& direction) {
    float theta = std::acos(std::clamp(direction[2], -1.0f, 1.0f));
    const float phi = -std::atan2(direction[1], direction[0]) + halfPi;
    const float cospsi = std::sin(theta) * std::sin(a.theta) * std::cos(a.phi - phi) + std::cos(theta) * std::cos(a.theta);
    const float gamma = std::acos(std::clamp(cospsi, -1.0f, 1.0f));
    theta = std::min(theta, halfPi - 0.001f);
    if (settings.type == SkyType::Hosek) {
        const Vector xyz{hosekInternal(a.config[0], theta, gamma) * a.radiance[0], hosekInternal(a.config[1], theta, gamma) * a.radiance[1],
                         hosekInternal(a.config[2], theta, gamma) * a.radiance[2]};
        const Vector rgb = xyzToRgb(xyz);
        const float scale = twoPi / 683.0f;
        return {rgb[0] * scale, rgb[1] * scale, rgb[2] * scale};
    }
    const float x = a.radiance[1] * perez(a.config[1], theta, gamma);
    const float y = a.radiance[2] * perez(a.config[2], theta, gamma);
    const float luminance = a.radiance[0] * perez(a.config[0], theta, gamma);
    if (y == 0.0f)
        return xyzToRgb({0.0f, luminance, 0.0f});
    return xyzToRgb({x / y * luminance, luminance, luminance == 0.0f ? 0.0f : (1.0f - x - y) / y * luminance});
}

//! Cycles' Nishita data: the look-up table of the sky, and the two sun pixels the disc blends between.
struct Nishita {
    std::vector<float> lut;
    Vector bottom{};
    Vector top{};
    Vector sun{};
    float elevation = 0.0f;
    float rotation = 0.0f;
    float diameter = 0.0f;
    float earthAngle = 0.0f;
};

//! Cycles' SkyTextureNode::simplify_settings: the elevation wraps into [-pi/2, pi/2] and the rotation turns into the shading convention.
std::pair<float, float> simplifySun(const float sunElevation, const float sunRotation) {
    float elevation = std::fmod(sunElevation, twoPi);
    float rotation = sunRotation;
    if (std::abs(elevation) >= pi)
        elevation -= std::copysign(2.0f, elevation) * pi;
    if (elevation >= halfPi || elevation <= -halfPi) {
        elevation = std::copysign(pi, elevation) - elevation;
        rotation += pi;
    }
    rotation = std::fmod(rotation, twoPi);
    if (rotation < 0.0f)
        rotation += twoPi;
    return {elevation, twoPi - rotation};
}

Nishita precomputeNishita(const SkySettings& settings) {
    Nishita n;
    const auto [elevation, rotation] = simplifySun(settings.sunElevation, settings.sunRotation);
    n.elevation = elevation;
    n.rotation = rotation;
    n.diameter = settings.sunDisc ? std::max(settings.sunSize, 0.0005f) : -1.0f;
    n.lut.resize(std::size_t{lutWidth} * lutHeight * 3);
    const bool multiple = settings.type == SkyType::MultipleScattering;
    if (multiple) {
        SKY_multiple_scattering_precompute_texture(n.lut.data(), 3, lutWidth, lutHeight, elevation, settings.altitude, settings.airDensity,
                                                   settings.aerosolDensity, settings.ozoneDensity);
        SKY_multiple_scattering_precompute_sun(elevation, std::max(settings.sunSize, 0.0005f), settings.altitude, settings.airDensity, settings.aerosolDensity,
                                               settings.ozoneDensity, n.bottom.data(), n.top.data());
    } else {
        SKY_single_scattering_precompute_texture(n.lut.data(), 3, lutWidth, lutHeight, elevation, settings.altitude, settings.airDensity,
                                                 settings.aerosolDensity, settings.ozoneDensity);
        SKY_single_scattering_precompute_sun(elevation, std::max(settings.sunSize, 0.0005f), settings.altitude, settings.airDensity, settings.aerosolDensity,
                                             n.bottom.data(), n.top.data());
    }
    n.earthAngle = -SKY_earth_intersection_angle(settings.altitude);
    n.sun = {-std::cos(elevation) * std::sin(rotation), std::cos(elevation) * std::cos(rotation), std::sin(elevation)};
    return n;
}

//! The look-up table with linear filtering and clamped edges, as Cycles' image with EXTENSION_EXTEND; row 0 is the bottom.
Vector lutSample(const Nishita& n, const float u, const float v) {
    const float x = std::clamp(u * lutWidth - 0.5f, 0.0f, static_cast<float>(lutWidth - 1));
    const float y = std::clamp(v * lutHeight - 0.5f, 0.0f, static_cast<float>(lutHeight - 1));
    const std::uint32_t x0 = static_cast<std::uint32_t>(x);
    const std::uint32_t y0 = static_cast<std::uint32_t>(y);
    const std::uint32_t x1 = std::min(x0 + 1, lutWidth - 1);
    const std::uint32_t y1 = std::min(y0 + 1, lutHeight - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    Vector result{};
    for (std::size_t c = 0; c < 3; ++c) {
        const auto at = [&](const std::uint32_t px, const std::uint32_t py) { return n.lut[(std::size_t{py} * lutWidth + px) * 3 + c]; };
        const float bottom = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
        const float top = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
        result[c] = bottom + (top - bottom) * fy;
    }
    return result;
}

//! Cycles' sky_radiance_nishita.
Vector nishitaRadiance(const SkySettings& settings, const Nishita& n, const Vector& direction) {
    const float theta = std::acos(std::clamp(direction[2], -1.0f, 1.0f));
    const float phi = std::atan2(direction[1], direction[0]);
    const float dirElevation = halfPi - theta;
    Vector xyz{};
    const float halfAngular = n.diameter * 0.5f;
    const float sunAngle = angleBetween(direction, n.sun);
    if (n.diameter >= 0.0f && sunAngle < halfAngular && dirElevation > n.earthAngle) {
        const float y = (dirElevation - n.elevation) / n.diameter + 0.5f;
        const float ratio = sunAngle / halfAngular;
        const float limb = 1.0f - 0.6f * (1.0f - std::sqrt(1.0f - ratio * ratio));
        for (std::size_t c = 0; c < 3; ++c)
            xyz[c] = (n.bottom[c] + (n.top[c] - n.bottom[c]) * y) * settings.sunIntensity * limb;
    }
    float x = (-phi - halfPi + n.rotation) / twoPi;
    x -= std::floor(x);
    const float v = std::copysign(std::sqrt(std::abs(dirElevation) * 2.0f / pi), dirElevation) * 0.5f + 0.5f;
    const Vector sky = lutSample(n, x, v);
    return xyzToRgb({xyz[0] + sky[0], xyz[1] + sky[1], xyz[2] + sky[2]});
}

//! The direction of a point of the table, as an Environment Texture looks it up: u = 0.5 - atan2(y, x) / 2 pi, and v from the zenith at the top row.
Vector tableDirection(const float column, const float row) {
    const float phi = (0.5f - column / tableWidth) * twoPi;
    const float theta = row / tableHeight * pi;
    return {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
}

}

SkySettings readSkySettings(Reader& reader, const Object& node) {
    SkySettings settings;
    const std::string type = reader.text(node, "sky_type");
    if (type == "PREETHAM")
        settings.type = SkyType::Preetham;
    else if (type == "HOSEK_WILKIE")
        settings.type = SkyType::Hosek;
    else if (type == "SINGLE_SCATTERING")
        settings.type = SkyType::SingleScattering;
    settings.sunDisc = reader.flag(node, "sun_disc");
    settings.sunSize = reader.number(node, "sun_size");
    settings.sunIntensity = reader.number(node, "sun_intensity");
    settings.sunElevation = reader.number(node, "sun_elevation");
    settings.sunRotation = reader.number(node, "sun_rotation");
    settings.altitude = reader.number(node, "altitude");
    settings.airDensity = reader.number(node, "air_density");
    settings.aerosolDensity = reader.number(node, "aerosol_density");
    settings.ozoneDensity = reader.number(node, "ozone_density");
    const Values direction = reader.values(node, "sun_direction");
    for (std::size_t i = 0; i < 3; ++i)
        settings.sunDirection[i] = direction.values[i];
    settings.turbidity = reader.number(node, "turbidity");
    settings.groundAlbedo = reader.number(node, "ground_albedo");
    return settings;
}

Scene::Texture makeSkyTexture(const SkySettings& settings, std::string name) {
    const bool nishita = settings.type == SkyType::SingleScattering || settings.type == SkyType::MultipleScattering;
    const Analytic analytic = settings.type == SkyType::Hosek ? precomputeHosek(settings) : precomputePreetham(settings);
    const Nishita scattering = nishita ? precomputeNishita(settings) : Nishita{};
    const auto radiance = [&](const Vector& direction) {
        return nishita ? nishitaRadiance(settings, scattering, direction) : analyticRadiance(settings, analytic, direction);
    };
    //! A texel within this angle of the sun centre may overlap the disc: the disc radius and a texel diagonal at the equator.
    const float reach = scattering.diameter * 0.5f + 2.0f * twoPi / tableWidth;
    std::vector<float> values(std::size_t{tableWidth} * tableHeight * 4);
    parallelFor(std::size_t{tableHeight} * minimumChunk, [&](std::size_t, const std::size_t begin, const std::size_t end) {
        for (std::size_t row = begin / minimumChunk; row < end / minimumChunk; ++row)
            for (std::uint32_t column = 0; column < tableWidth; ++column) {
                const float rowCentre = static_cast<float>(row) + 0.5f;
                const float columnCentre = static_cast<float>(column) + 0.5f;
                const Vector centre = tableDirection(columnCentre, rowCentre);
                Vector sum{};
                std::uint32_t count = 1;
                if (nishita && scattering.diameter >= 0.0f && angleBetween(centre, scattering.sun) < reach) {
                    count = sunSamples * sunSamples;
                    for (std::uint32_t i = 0; i < count; ++i) {
                        const Vector sample = radiance(tableDirection(static_cast<float>(column) + (static_cast<float>(i % sunSamples) + 0.5f) / sunSamples,
                                                                      static_cast<float>(row) + (static_cast<float>(i / sunSamples) + 0.5f) / sunSamples));
                        for (std::size_t c = 0; c < 3; ++c)
                            sum[c] += sample[c];
                    }
                } else {
                    sum = radiance(centre);
                }
                float* texel = values.data() + (row * tableWidth + column) * 4;
                for (std::size_t c = 0; c < 3; ++c)
                    texel[c] = sum[c] / static_cast<float>(count);
                texel[3] = 1.0f;
            }
    });
    Scene::Texture texture;
    texture.name = std::move(name);
    texture.format = NoorRHI::ImageFormat::Rgba32Float;
    texture.width = tableWidth;
    texture.height = tableHeight;
    texture.data.resize(values.size() * sizeof(float));
    std::memcpy(texture.data.data(), values.data(), texture.data.size());
    texture.levels.push_back({.offset = 0, .size = texture.data.size(), .width = tableWidth, .height = tableHeight});
    return texture;
}

}
