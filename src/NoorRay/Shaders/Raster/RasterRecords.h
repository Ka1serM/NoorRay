#pragma once

#include "NoorRHI/Shaders/ShaderTypes.h"
#include "NoorRay/Shaders/Common/Shared.h"
#if !defined(__SLANG__)
#include "NoorRay/Shaders/Common/CameraRecords.h"
#include "NoorRay/Shaders/Common/GpuRecords.h"
#endif

NR_NAMESPACE_BEGIN

//! The root arguments of one visibility draw: the triangles of instance get the ids firstTriangle + primitive.
struct VisibilityArguments {
    Ptr<SceneRecord> scene;
    Ptr<CameraRecord> camera;
    float2 jitter;
    uint instance;
    uint firstTriangle;
};

NR_NAMESPACE_END

#if !defined(__SLANG__)
#include <cstddef>
static_assert(sizeof(NoorRay::Shaders::VisibilityArguments) == 32);
static_assert(offsetof(NoorRay::Shaders::VisibilityArguments, jitter) == 16);
#endif
