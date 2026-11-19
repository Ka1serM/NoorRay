#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"

NR_NAMESPACE_BEGIN

static const uint cameraTypePerspective = 0u;
static const uint cameraTypeOrthographic = 1u;
static const uint cameraTypeFisheye = 2u;
static const uint cameraTypeRealistic = 3u;

//! One spherical surface in lens space, where the film is at z = 0 and the lens toward -z.
struct LensElementRecord {
    //! Curvature radius (0 for the stop), vertex z, aperture radius.
    float4 geometry;
    //! IOR and inverse Abbe number on the film side.
    float4 medium;
};

//! Each transform is three rows of a 3x4 matrix.
struct CameraRecord {
    float4 cameraToWorld[3];
    float4 worldToCamera[3];
    float4 previousCameraToWorld[3];
    float4 previousWorldToCamera[3];
    //! Film width and height in metres, fitted to the image aspect, then shift x and y.
    float4 sensor;
    //! Focal length, orthographic height, focus distance, pixel spread angle.
    float4 projection;
    //! Radius, blade count, blade rotation, cos^4 normalization.
    float4 aperture;
    float4 clip;
    //! Type, lens element count, image width, image height.
    uint4 meta;
    Ptr<LensElementRecord> lensElements;
    //! The z of the front vertex, which maps to the camera origin.
    float lensFront;
    float rearRadius;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
static_assert(sizeof(NoorRay::Shaders::LensElementRecord) == 32);
static_assert(sizeof(NoorRay::Shaders::CameraRecord) == 288);
#endif
