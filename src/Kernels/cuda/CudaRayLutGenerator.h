#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

#include "Scene/SceneTypes.h"

void generateCudaRayLut(
    const RealisticCameraSettings& settings,
    RayLutEntry* deviceOutput,
    uint32_t width,
    uint32_t height,
    cudaStream_t stream);
