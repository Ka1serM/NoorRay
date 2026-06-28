#pragma once

#if defined(NR_BACKEND_CUDA)
#define NR_CUDA_ACTIVE 1
#endif

#if defined(__CUDA_ARCH__)
#define NR_GPU_DEVICE_COMPILE 1
#endif

#if defined(__CUDACC__)
#define NR_CPU_GPU __host__ __device__
#define NR_GPU __device__
#define NR_GPU_KERNEL __global__
#define NR_GPU_CODE 1
#else
#define NR_CPU_GPU
#define NR_GPU
#define NR_GPU_KERNEL
#endif

#if defined(__CUDACC__)
#define NR_GPU_LAUNCH_IDX (blockIdx.x * blockDim.x + threadIdx.x)
#define NR_GPU_OPTIX_LAUNCH_ID optixGetLaunchIndex().x
#endif
