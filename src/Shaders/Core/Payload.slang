#ifndef PAYLOAD_SLANG
#define PAYLOAD_SLANG

#include "../Shared.h"

struct Payload
{
    float3 Attenuation; uint Flags;
    float3 NextDirection; uint RngState;
    float LastBsdfPdf;

    [mutating]
    void initialize(uint rngState)
    {
        Attenuation = float3(1.0);
        Flags = 0u;
        NextDirection = float3(0.0);
        RngState = rngState;
        LastBsdfPdf = 0.0;
    }
};

#endif // PAYLOAD_SLANG
