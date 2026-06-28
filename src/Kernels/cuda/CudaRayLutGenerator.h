#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

#include "Camera/RealisticCamera.h"

void generateCudaRayLut(
    const RealisticCamera* camera,
    RayLutEntry* deviceOutput,
    uint32_t width,
    uint32_t height,
    cudaStream_t stream);
