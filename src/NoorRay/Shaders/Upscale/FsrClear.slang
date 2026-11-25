import NoorRHI;

//! The layout of NoorRay::Render::FsrClearArguments: a storage image, or a buffer when count is not 0.
struct FsrClearArguments {
    float4 value;
    StorageTextureHandle image;
    uint width;
    uint height;
    uint count;
    AtomicPtr<uint> buffer;
};

[shader("compute")]
[numthreads(8, 8, 1)]
void fsrClear(uint3 position : SV_DispatchThreadID) {
    const FsrClearArguments arguments = rootArguments<FsrClearArguments>();
    if (arguments.count != 0) {
        const uint index = position.y * arguments.width + position.x;
        if (position.x < arguments.width && index < arguments.count)
            atomicStore(arguments.buffer + index, asuint(arguments.value.x));
        return;
    }
    if (position.x < arguments.width && position.y < arguments.height)
        arguments.image[position.xy] = arguments.value;
}
