import NoorRHI;
import Raster.Records;
import Common.Math;
import Scene.Camera;
import Scene.Gaussians;

static const uint noSplatDigit = splatDigits;
//! The low-pass filter of 3DGS in square pixels, which keeps every splat at least about a pixel wide.
static const float splatDilation = 0.3;
//! Keeps the half-precision pixel axes finite for a Gaussian that touches the camera.
static const float splatMaxAxis = 16384.0;
static const uint splatScanItems = splatScanTile / splatGroupSize;

groupshared uint4 splatRankScan[splatGroupSize];
groupshared uint splatSumScan[splatGroupSize];

uint getSplatBlock(uint3 group) { return group.y * splatRowGroups + group.x; }

uint packHalves(float a, float b) { return f32tof16(a) | (f32tof16(b) << 16); }

float2 unpackHalves(uint word) { return float2(f16tof32(word & 0xffffu), f16tof32(word >> 16)); }

void resetSplatCounters(SplatArguments arguments) {
    atomicStore(arguments.counters + splatCounterVisible, 0u);
    atomicStore(arguments.counters + splatCounterMinDepth, 0xffffffffu);
    atomicStore(arguments.counters + splatCounterMaxDepth, 0u);
}

//! The cloud that owns a global Gaussian index; scenes have few clouds.
uint findCloud(SceneRecord scene, uint cloudCount, uint index) {
    uint result = 0;
    for (uint cloud = 1; cloud < cloudCount; ++cloud)
        if (scene.gaussianClouds[cloud].firstInstance <= index)
            result = cloud;
    return result;
}

//! The camera-space axis of one standard deviation along a local axis of the Gaussian.
float3 cameraAxis(CameraRecord camera, GaussianCloudRecord cloud, Gaussian gaussian, float3 axis) {
    return transformVector(camera.worldToCamera, transformVector(cloud.objectToWorld, rotateQuaternion(gaussian.rotation, axis)));
}

//! The pixel offset (y down) of a camera-space offset at the camera-space point local, to first order.
float2 projectAxis(CameraRecord camera, float3 local, float3 axis) {
    const float2 pixels = float2(camera.meta.zw);
    if (camera.meta.x == cameraTypeOrthographic) {
        const float scale = camera.sensor.y / camera.projection.y;
        return float2(axis.x, -axis.y) * scale * pixels / camera.sensor.xy;
    }
    const float depth = -local.z;
    const float2 focal = camera.projection.x * pixels / camera.sensor.xy;
    //! 3DGS clamps the tangent to 1.3 times the half field of view, so the Jacobian of a Gaussian far outside the image stays bounded.
    const float2 limit = 1.3 * 0.5 * camera.sensor.xy / camera.projection.x;
    const float2 tangent = clamp(local.xy / depth, -limit, limit);
    return float2(focal.x / depth * (axis.x + tangent.x * axis.z), -focal.y / depth * (axis.y + tangent.y * axis.z));
}

//! Projects one Gaussian to its screen ellipse, cut at alpha 1/255, and appends it to the visible list when the ellipse meets the image.
void projectSplat(SplatArguments arguments, uint index) {
    const SceneRecord scene = arguments.scene[0];
    const CameraRecord camera = arguments.camera[0];
    const GaussianCloudRecord cloud = scene.gaussianClouds[findCloud(scene, arguments.cloudCount, index)];
    const uint local = index - cloud.firstInstance;
    const float4 positionOpacity = cloud.positionOpacity[local];
    const float cutoff = gaussianCutoffSquared(positionOpacity.w);
    const float3 world = transformPoint(cloud.objectToWorld, positionOpacity.xyz);
    const float3 view = transformPoint(camera.worldToCamera, world);
    const float depth = -view.z;
    if (cutoff <= 0.0 || depth <= camera.clip.x || depth >= max(camera.clip.y, camera.clip.x * 2.0))
        return;
    const float4 clip = projectToClip(camera, world, float2(0.5, 0.5));
    const float3 ndc = clip.xyz / clip.w;
    //! A cheap bound before the covariance: 3DGS limits the tangent, so a centre far outside the image is never visible.
    if (any(abs(ndc.xy) > 1.3 + 1.0))
        return;
    const Gaussian gaussian = loadGaussian(cloud, local, positionOpacity);
    const float2 b0 = projectAxis(camera, view, cameraAxis(camera, cloud, gaussian, float3(gaussian.scale.x, 0.0, 0.0)));
    const float2 b1 = projectAxis(camera, view, cameraAxis(camera, cloud, gaussian, float3(0.0, gaussian.scale.y, 0.0)));
    const float2 b2 = projectAxis(camera, view, cameraAxis(camera, cloud, gaussian, float3(0.0, 0.0, gaussian.scale.z)));
    const float a = b0.x * b0.x + b1.x * b1.x + b2.x * b2.x + splatDilation;
    const float b = b0.x * b0.y + b1.x * b1.y + b2.x * b2.y;
    const float c = b0.y * b0.y + b1.y * b1.y + b2.y * b2.y + splatDilation;
    const float middle = 0.5 * (a + c);
    const float radius = sqrt(max(0.25 * (a - c) * (a - c) + b * b, 0.0));
    const float major = middle + radius;
    const float minor = max(middle - radius, 1e-4);
    const float2 direction = abs(b) > 1e-12 ? normalize(float2(b, major - a)) : (a >= c ? float2(1.0, 0.0) : float2(0.0, 1.0));
    const float2 axis1 = direction * min(sqrt(cutoff * major), splatMaxAxis);
    const float2 axis2 = float2(-direction.y, direction.x) * min(sqrt(cutoff * minor), splatMaxAxis);
    const float2 pixels = float2(camera.meta.zw);
    const float2 center = float2(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * pixels;
    //! The bounding box of the ellipse, with a pixel of margin for the jitter.
    const float2 extent = sqrt(axis1 * axis1 + axis2 * axis2) + 1.0;
    if (any(center < -extent) || any(center > pixels + extent))
        return;
    const float3 cameraObject = transformPoint(cloud.worldToObject, cameraPosition(camera));
    const float3 color = decodeGaussianColor(evaluateGaussianSh(cloud, local, gaussian.mean - cameraObject));
    arguments.splats[index * 2] = asuint(float4(ndc, gaussian.opacity));
    arguments.splats[index * 2 + 1] =
        uint4(packHalves(axis1.x, axis1.y), packHalves(axis2.x, axis2.y), packHalves(color.r, color.g), packHalves(color.b, 0.0));
    const uint slot = atomicAdd(arguments.counters + splatCounterVisible, 1u);
    arguments.keys[slot] = asuint(depth);
    arguments.values[slot] = index;
    atomicMin(arguments.counters + splatCounterMinDepth, asuint(depth));
    atomicMax(arguments.counters + splatCounterMaxDepth, asuint(depth));
}

//! Turns the counters into the sort sizes, the indirect dispatches and the instanced draw, then resets them for the next frame.
void prepareSplats(SplatArguments arguments) {
    const uint visible = min(atomicLoad(arguments.counters + splatCounterVisible), arguments.count);
    const uint blocks = (visible + splatGroupSize - 1) / splatGroupSize;
    const uint partials = (blocks * splatDigits + splatScanTile - 1) / splatScanTile;
    const float logMin = visible == 0 ? 0.0 : log(asfloat(atomicLoad(arguments.counters + splatCounterMinDepth)));
    const float logMax = visible == 0 ? 0.0 : log(asfloat(atomicLoad(arguments.counters + splatCounterMaxDepth)));
    const float range = logMax - logMin;
    arguments.state[splatStateVisible] = visible;
    arguments.state[splatStateBlocks] = blocks;
    arguments.state[splatStatePartials] = partials;
    arguments.state[splatStateLogMin] = asuint(logMin);
    arguments.state[splatStateScale] = asuint(range > 0.0 ? 65535.0 / range : 0.0);
    DispatchSize blockDispatch;
    blockDispatch.x = min(blocks, splatRowGroups);
    blockDispatch.y = (blocks + splatRowGroups - 1) / splatRowGroups;
    blockDispatch.z = 1;
    arguments.dispatches[splatDispatchBlocks] = blockDispatch;
    DispatchSize scanDispatch;
    scanDispatch.x = partials;
    scanDispatch.y = 1;
    scanDispatch.z = 1;
    arguments.dispatches[splatDispatchScan] = scanDispatch;
    DrawIndexedArgs draw;
    draw.indexCount = 6;
    draw.instanceCount = visible;
    draw.firstIndex = 0;
    draw.vertexOffset = 0;
    draw.firstInstance = 0;
    arguments.draw[0] = draw;
    resetSplatCounters(arguments);
}

uint4 getSplatDigitMask(uint digit) {
    uint4 mask = uint4(0u);
    if (digit < noSplatDigit)
        mask[digit >> 2] = 1u << ((digit & 3u) * 8u);
    return mask;
}

uint getSplatDigitCount(uint4 counts, uint digit) { return (counts[digit >> 2] >> ((digit & 3u) * 8u)) & 0xffu; }

//! Inclusive per-digit counts over the block; 8-bit lanes hold them because the block has 128 threads.
uint4 rankSplatDigits(uint thread, uint digit) {
    splatRankScan[thread] = getSplatDigitMask(digit);
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < splatGroupSize; offset <<= 1) {
        const uint4 previous = thread >= offset ? splatRankScan[thread - offset] : uint4(0u);
        GroupMemoryBarrierWithGroupSync();
        splatRankScan[thread] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return splatRankScan[thread];
}

//! Back to front: the farthest splat gets key 0. The first pass reads the depth bits that projectSplat wrote.
uint readSplatKey(SplatArguments arguments, uint index) {
    const uint stored = arguments.keys[index];
    if (arguments.shift != 0)
        return stored;
    const float logMin = asfloat(arguments.state[splatStateLogMin]);
    const float scale = asfloat(arguments.state[splatStateScale]);
    return 65535u - uint(clamp((log(asfloat(stored)) - logMin) * scale + 0.5, 0.0, 65535.0));
}

uint getSplatDigit(SplatArguments arguments, uint key) { return (key >> arguments.shift) & (splatDigits - 1u); }

void countSplatBlock(SplatArguments arguments, uint block, uint thread) {
    const uint visible = arguments.state[splatStateVisible];
    const uint blocks = arguments.state[splatStateBlocks];
    const uint index = block * splatGroupSize + thread;
    rankSplatDigits(thread, index < visible ? getSplatDigit(arguments, readSplatKey(arguments, index)) : noSplatDigit);
    if (thread < splatDigits && block < blocks)
        arguments.histogram[thread * blocks + block] = getSplatDigitCount(splatRankScan[splatGroupSize - 1], thread);
}

uint scanSplatGroup(uint thread, uint value) {
    splatSumScan[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < splatGroupSize; offset <<= 1) {
        const uint previous = thread >= offset ? splatSumScan[thread - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        splatSumScan[thread] += previous;
        GroupMemoryBarrierWithGroupSync();
    }
    return splatSumScan[thread];
}

//! The first level of the scan: the sum of each tile of the histogram, read coalesced.
void reduceSplatTile(SplatArguments arguments, uint tile, uint thread) {
    const uint total = arguments.state[splatStateBlocks] * splatDigits;
    uint sum = 0;
    for (uint item = 0; item < splatScanItems; ++item) {
        const uint index = tile * splatScanTile + item * splatGroupSize + thread;
        if (index < total)
            sum += arguments.histogram[index];
    }
    const uint tileSum = scanSplatGroup(thread, sum);
    if (thread == splatGroupSize - 1)
        arguments.partials[tile] = tileSum;
}

//! The exclusive scan of the tile sums in one workgroup; each thread scans a contiguous segment.
void scanSplatTiles(SplatArguments arguments, uint thread) {
    const uint total = arguments.state[splatStatePartials];
    const uint segment = (total + splatGroupSize - 1) / splatGroupSize;
    const uint begin = min(thread * segment, total);
    const uint end = min(begin + segment, total);
    uint sum = 0;
    for (uint index = begin; index < end; ++index)
        sum += arguments.partials[index];
    uint running = scanSplatGroup(thread, sum) - sum;
    for (uint index = begin; index < end; ++index) {
        const uint count = arguments.partials[index];
        arguments.partials[index] = running;
        running += count;
    }
}

//! The last level: the exclusive scan inside each tile, offset by the scanned tile sum.
void downsweepSplatTile(SplatArguments arguments, uint tile, uint thread) {
    const uint total = arguments.state[splatStateBlocks] * splatDigits;
    const uint begin = tile * splatScanTile + thread * splatScanItems;
    uint counts[splatScanItems];
    uint sum = 0;
    for (uint item = 0; item < splatScanItems; ++item) {
        counts[item] = begin + item < total ? arguments.histogram[begin + item] : 0u;
        sum += counts[item];
    }
    uint running = arguments.partials[tile] + scanSplatGroup(thread, sum) - sum;
    for (uint item = 0; item < splatScanItems; ++item) {
        if (begin + item < total)
            arguments.histogram[begin + item] = running;
        running += counts[item];
    }
}

void scatterSplatBlock(SplatArguments arguments, uint block, uint thread) {
    const uint visible = arguments.state[splatStateVisible];
    const uint blocks = arguments.state[splatStateBlocks];
    const uint index = block * splatGroupSize + thread;
    const bool valid = index < visible;
    const uint key = valid ? readSplatKey(arguments, index) : 0u;
    const uint digit = valid ? getSplatDigit(arguments, key) : noSplatDigit;
    const uint4 ranks = rankSplatDigits(thread, digit);
    if (!valid)
        return;
    const uint target = arguments.histogram[digit * blocks + block] + getSplatDigitCount(ranks, digit) - 1;
    arguments.sortedKeys[target] = key;
    arguments.sortedValues[target] = arguments.values[index];
}

struct SplatVaryings {
    float4 position : SV_Position;
    float2 offset : SPLAT_OFFSET;
    nointerpolation float4 color : SPLAT_COLOR;
};

//! One instance per visible splat in back-to-front order and four corners each; the quad spans the ellipse axes at the jittered centre.
SplatVaryings projectSplatVertex(SplatArguments arguments, uint corner, uint instance) {
    const uint splat = arguments.values[instance];
    const float4 center = asfloat(arguments.splats[splat * 2]);
    const uint4 packed = arguments.splats[splat * 2 + 1];
    const float2 offset = float2((corner & 1u) != 0 ? 1.0 : -1.0, (corner & 2u) != 0 ? 1.0 : -1.0);
    const float2 pixels = float2(arguments.camera[0].meta.zw);
    const float2 pixelOffset = offset.x * unpackHalves(packed.x) + offset.y * unpackHalves(packed.y) + (float2(0.5, 0.5) - arguments.jitter);
    SplatVaryings output;
    output.position = float4(center.xy + pixelOffset * float2(2.0, -2.0) / pixels, center.z, 1.0);
    output.offset = offset;
    output.color = float4(unpackHalves(packed.z), unpackHalves(packed.w).x, center.w);
    return output;
}

//! Premultiplied linear radiance; the quad corners sit at the cutoff, so alpha is opacity * exp(-cutoff / 2 * r^2).
float4 shadeSplat(SplatVaryings input) {
    const float opacity = input.color.w;
    const float alpha = min(opacity * exp(-0.5 * gaussianCutoffSquared(opacity) * dot(input.offset, input.offset)), 0.99);
    if (alpha < 1.0 / 255.0)
        discard;
    return float4(input.color.rgb * alpha, alpha);
}
