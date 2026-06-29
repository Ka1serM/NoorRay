#pragma once

#include <cuda_runtime_api.h>

#include "Raytracing/SceneData.h"

void launchShade(const KernelParams& params, cudaStream_t stream);
