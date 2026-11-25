#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "NoorRay/Render/RendererSettings.h"

namespace NoorRay::Render {

//! The quality modes of FSR 3 and DLSS; the render scale of each is in upscaleQualityScales.
enum class UpscaleQuality : std::uint8_t { Native, Quality, Balanced, Performance, UltraPerformance };

constexpr std::array<float, 5> upscaleQualityScales{1.0f, 1.0f / 1.5f, 1.0f / 1.7f, 0.5f, 1.0f / 3.0f};

//! The render settings of the Blender add-on and of the app: the defaults are those of the add-on and of Cycles, so a scene renders the same in both.
struct RenderOptions {
    //! The accumulated samples of a final render, or the realtime frames of a viewport before it stops; zero renders without a limit.
    std::uint32_t samples = 256;
    //! NRD (RELAX) denoises the accumulated film; OpenDLSS-RR denoises and upscales frames that it does not accumulate.
    Denoiser denoiser = Denoiser::OpenDlssRr;
    //! False keeps each frame apart: no denoiser history.
    bool temporal = true;
    //! A final render accumulates and waits for its kernels; the viewport renders each frame on its own with the kernels it has.
    bool finalRender = false;
    //! Cycles' light paths: the bounces after the camera hit, the volume scattering events and the direct and indirect clamps, 0 for none.
    std::uint32_t maxBounces = 12;
    std::uint32_t volumeBounces = 0;
    std::uint32_t transparentBounces = 8;
    std::uint32_t diffuseBounces = 4;
    std::uint32_t glossyBounces = 4;
    std::uint32_t transmissionBounces = 12;
    //! Cycles' Filter Glossy, 0 for none.
    float filterGlossy = 1.0f;
    //! Cycles' Film Exposure, a linear factor.
    float exposure = 1.0f;
    float clampDirect = 0.0f;
    float clampIndirect = 10.0f;
    //! Cycles' Pixel Filter width in pixels.
    float filterWidth = 1.5f;
    //! Cycles' Light Tree; false picks the lights by their power alone.
    bool lightTree = true;
    //! Learns which lights are occluded where, so that fewer shadow rays miss.
    bool visibilityCache = true;
    //! Off leaves the Gaussians out of the traced rays; emissive shows their colour; scattering lights them as isotropic scatterers.
    GaussianShading gaussianShading = GaussianShading::Emissive;
    //! Cycles' Seed, with the frame added for an animated seed.
    std::uint32_t seed = 0;
    //! The realtime settings: ReSTIR DI or PT, the SHaRC radiance cache, and FSR with its sharpness.
    Reuse reuse = Reuse::Off;
    bool radianceCache = true;
    bool upscale = true;
    //! The render resolution over the display resolution of FSR and DLSS-RR.
    float renderScale = upscaleQualityScales[static_cast<std::size_t>(UpscaleQuality::Quality)];
    float sharpness = 0.2f;

    //! The path tracer settings of these options; the display settings stay at their defaults.
    RendererSettings toRendererSettings() const;
};

}
