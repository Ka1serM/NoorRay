#include <cstdio>
#include <cuda_runtime.h>

// Include our LUT header first (sets OPENPBR_USE_TEXTURE_LUTS = 1, etc.).
// For the data-extraction below we temporarily switch to array mode.
#include "OpenPbrLuts.h"

// ──────────────────────────────────────────────────────────────────────────
// Host-side data extraction — we need the raw arrays which are only included
// when OPENPBR_USE_TEXTURE_LUTS == 0.  Undefine it here and provide the
// minimal interop glue that those data headers expect.
// ──────────────────────────────────────────────────────────────────────────
#undef OPENPBR_USE_TEXTURE_LUTS
#define OPENPBR_USE_TEXTURE_LUTS 0

// Minimal interop for host-side data arrays.
#include <cstdint>
#define OPENPBR_UINT32 std::uint32_t
#define OPENPBR_UINT16 std::uint16_t
#define OPENPBR_ENERGY_TABLES_USE_UINT16 1
#define OpenPBR_EnergyTableElement OPENPBR_UINT16
#define OPENPBR_CONSTEXPR_GLOBAL static constexpr

// vec3 needed for the LTC data.
struct Vec3f { float x, y, z; };
#define vec3 Vec3f
inline Vec3f make_vec3(float x, float y, float z) { return {x, y, z}; }

#include "../../external/openpbr-bsdf/openpbr_data_constants.h"
#include "../../external/openpbr-bsdf/impl/data/openpbr_energy_arrays.h"
#include "../../external/openpbr-bsdf/impl/data/openpbr_ltc_array.h"

// ──────────────────────────────────────────────────────────────────────────
// Device-side symbol that the macros reference.
// ──────────────────────────────────────────────────────────────────────────
__device__ cudaTextureObject_t openpbrLuts[8];

// ──────────────────────────────────────────────────────────────────────────
// Texture creation helpers.
// ──────────────────────────────────────────────────────────────────────────

static cudaTextureObject_t makeTex2D_r32f(const float* data, int w, int h)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray_t arr{};
    cudaMallocArray(&arr, &desc, w, h);
    cudaMemcpy2DToArray(arr, 0, 0, data,
                        w * sizeof(float), w * sizeof(float), h,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;

    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode     = cudaFilterModeLinear;
    td.readMode       = cudaReadModeElementType;
    td.normalizedCoords = 1;

    cudaTextureObject_t obj{};
    cudaCreateTextureObject(&obj, &rd, &td, nullptr);
    return obj;
}

static cudaTextureObject_t makeTex2D_rgba32f(const float* data, int w, int h)
{
    cudaChannelFormatDesc desc =
        cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaArray_t arr{};
    cudaMallocArray(&arr, &desc, w, h);
    cudaMemcpy2DToArray(arr, 0, 0, data,
                        w * 4 * sizeof(float), w * 4 * sizeof(float), h,
                        cudaMemcpyHostToDevice);

    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;

    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode     = cudaFilterModeLinear;
    td.readMode       = cudaReadModeElementType;
    td.normalizedCoords = 1;

    cudaTextureObject_t obj{};
    cudaCreateTextureObject(&obj, &rd, &td, nullptr);
    return obj;
}

static cudaTextureObject_t makeTex3D_r32f(const float* data, int d)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(d, d, d);
    cudaArray_t arr{};
    cudaMalloc3DArray(&arr, &desc, extent);

    cudaMemcpy3DParms cp{};
    cp.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(data), d * sizeof(float), d, d);
    cp.dstArray = arr;
    cp.extent = extent;
    cp.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&cp);

    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;

    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.addressMode[2] = cudaAddressModeClamp;
    td.filterMode     = cudaFilterModeLinear;
    td.readMode       = cudaReadModeElementType;
    td.normalizedCoords = 1;

    cudaTextureObject_t obj{};
    cudaCreateTextureObject(&obj, &rd, &td, nullptr);
    return obj;
}

// ──────────────────────────────────────────────────────────────────────────
void uploadOpenPbrLuts()
{
    constexpr int N = OpenPBR_EnergyTableSize;
    auto toFloat = [](auto v) { return float(v) * (1.0f / 65535.0f); };

    cudaTextureObject_t hostLuts[8] = {};

    // 0. IdealDielectricEnergyComplement (3D, 32x32x32)
    {
        constexpr int sz = N * N * N;
        float* buf = new float[sz];
        for (int i = 0; i < sz; ++i)
            buf[i] = toFloat(OpenPBR_IdealDielectricEnergyComplement_Array[i]);
        hostLuts[OpenPBR_LutId_IdealDielectricEnergyComplement] = makeTex3D_r32f(buf, N);
        delete[] buf;
    }

    // 1. IdealDielectricAverageEnergyComplement (2D, 32x32)
    {
        constexpr int sz = N * N;
        float* buf = new float[sz];
        for (int i = 0; i < sz; ++i)
            buf[i] = toFloat(OpenPBR_IdealDielectricAverageEnergyComplement_Array[i]);
        hostLuts[OpenPBR_LutId_IdealDielectricAverageEnergyComplement] = makeTex2D_r32f(buf, N, N);
        delete[] buf;
    }

    // 2. IdealDielectricReflectionRatio (2D, 32x32)
    {
        constexpr int sz = N * N;
        float* buf = new float[sz];
        for (int i = 0; i < sz; ++i)
            buf[i] = toFloat(OpenPBR_IdealDielectricReflectionRatio_Array[i]);
        hostLuts[OpenPBR_LutId_IdealDielectricReflectionRatio] = makeTex2D_r32f(buf, N, N);
        delete[] buf;
    }

    // 3. OpaqueDielectricEnergyComplement (3D, 32x32x32)
    {
        constexpr int sz = N * N * N;
        float* buf = new float[sz];
        for (int i = 0; i < sz; ++i)
            buf[i] = toFloat(OpenPBR_OpaqueDielectricEnergyComplement_Array[i]);
        hostLuts[OpenPBR_LutId_OpaqueDielectricEnergyComplement] = makeTex3D_r32f(buf, N);
        delete[] buf;
    }

    // 4. OpaqueDielectricAverageEnergyComplement (2D, 32x32)
    {
        constexpr int sz = N * N;
        float* buf = new float[sz];
        for (int i = 0; i < sz; ++i)
            buf[i] = toFloat(OpenPBR_OpaqueDielectricAverageEnergyComplement_Array[i]);
        hostLuts[OpenPBR_LutId_OpaqueDielectricAverageEnergyComplement] = makeTex2D_r32f(buf, N, N);
        delete[] buf;
    }

    // 5. IdealMetalEnergyComplement (2D, 32x32)
    {
        constexpr int sz = N * N;
        float* buf = new float[sz];
        for (int i = 0; i < sz; ++i)
            buf[i] = toFloat(OpenPBR_IdealMetalEnergyComplement_Array[i]);
        hostLuts[OpenPBR_LutId_IdealMetalEnergyComplement] = makeTex2D_r32f(buf, N, N);
        delete[] buf;
    }

    // 6. IdealMetalAverageEnergyComplement (1D stored as thin 2D: Nx1)
    {
        float* buf = new float[N];
        for (int i = 0; i < N; ++i)
            buf[i] = toFloat(OpenPBR_IdealMetalAverageEnergyComplement_Array[i]);
        hostLuts[OpenPBR_LutId_IdealMetalAverageEnergyComplement] = makeTex2D_r32f(buf, N, 1);
        delete[] buf;
    }

    // 7. LTC (2D, 3 channels → stored as RGBA32F with .w = 1)
    {
        constexpr int sz = OpenPBR_LTCTableSize * OpenPBR_LTCTableSize;
        float* buf = new float[sz * 4];
        for (int i = 0; i < sz; ++i)
        {
            buf[i * 4 + 0] = OpenPBR_LTC_Array[i].x;
            buf[i * 4 + 1] = OpenPBR_LTC_Array[i].y;
            buf[i * 4 + 2] = OpenPBR_LTC_Array[i].z;
            buf[i * 4 + 3] = 1.0f;
        }
        hostLuts[OpenPBR_LutId_LTC] = makeTex2D_rgba32f(
            buf, OpenPBR_LTCTableSize, OpenPBR_LTCTableSize);
        delete[] buf;
    }

    cudaMemcpyToSymbol(openpbrLuts, hostLuts, sizeof(hostLuts));
}

void destroyOpenPbrLuts()
{
    cudaTextureObject_t hostCopy[8] = {};
    cudaMemcpyFromSymbol(hostCopy, openpbrLuts, sizeof(hostCopy));
    for (int i = 0; i < 8; ++i)
    {
        if (hostCopy[i])
        {
            cudaDestroyTextureObject(hostCopy[i]);
            hostCopy[i] = {};
        }
    }
    cudaMemcpyToSymbol(openpbrLuts, hostCopy, sizeof(hostCopy));
}
