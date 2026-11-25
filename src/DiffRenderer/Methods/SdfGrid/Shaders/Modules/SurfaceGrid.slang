#pragma once

#include "SurfaceRendering.h"

typealias Arguments = SurfaceRendering::Arguments;
typealias Grid = SurfaceRendering::Grid;

static const uint shCoefficients = SurfaceRendering::shCoefficients;
static const uint colourChannels = SurfaceRendering::colourChannels;
static const float pi = 3.14159265358979;

struct Random {
    uint state;

    __init(uint a, uint b) {
        state = hash(a ^ hash(b + 0x9e3779b9u));
    }

    static uint hash(uint value) {
        const uint state = value * 747796405u + 2891336453u;
        const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    [mutating]
    float next() {
        state = hash(state);
        return float(state >> 8) * (1.0 / 16777216.0);
    }
}

struct Cell {
    uint base;
    uint rowStride;
    uint sliceStride;
    float3 fraction;

    uint getCorner(uint corner) {
        return base + (corner & 1) + ((corner >> 1) & 1) * rowStride + (corner >> 2) * sliceStride;
    }

    float getWeight(uint corner) {
        const float3 bits = float3(float(corner & 1), float((corner >> 1) & 1), float(corner >> 2));
        const float3 weights = lerp(1.0 - fraction, fraction, bits);
        return weights.x * weights.y * weights.z;
    }

    //! The derivatives of getWeight along x, y and z, in grid units.
    float3 getSlope(uint corner) {
        const float3 bits = float3(float(corner & 1), float((corner >> 1) & 1), float(corner >> 2));
        const float3 weights = lerp(1.0 - fraction, fraction, bits);
        const float3 signs = bits * 2.0 - 1.0;
        return signs * float3(weights.y * weights.z, weights.x * weights.z, weights.x * weights.y);
    }
}

Cell getCell(Grid grid, float3 position) {
    const float3 last = float3(grid.size.xyz - 1);
    const float3 local = clamp((position - grid.origin.xyz) / grid.origin.w, float3(0.0), last);
    const uint3 node = min(uint3(local), grid.size.xyz - 2);
    Cell cell;
    cell.rowStride = grid.size.x;
    cell.sliceStride = grid.size.x * grid.size.y;
    cell.base = node.x + node.y * cell.rowStride + node.z * cell.sliceStride;
    cell.fraction = local - float3(node);
    return cell;
}

int3 getNodePosition(Grid grid, uint index) {
    return int3(int(index % grid.size.x), int((index / grid.size.x) % grid.size.y), int(index / (grid.size.x * grid.size.y)));
}

uint getNodeIndex(Grid grid, int3 node) {
    return uint(node.x) + uint(node.y) * grid.size.x + uint(node.z) * grid.size.x * grid.size.y;
}

bool isNode(Grid grid, int3 node) {
    return all(node >= 0) && all(node < int3(grid.size.xyz));
}

float sampleScalar(Ptr<float> values, Cell cell) {
    float value = 0.0;
    [ForceUnroll]
    for (uint corner = 0; corner < 8; ++corner)
        value += cell.getWeight(corner) * values[cell.getCorner(corner)];
    return value;
}

float3 sampleSlope(Ptr<float> values, Cell cell, float voxel) {
    float3 slope = float3(0.0);
    [ForceUnroll]
    for (uint corner = 0; corner < 8; ++corner)
        slope += cell.getSlope(corner) * values[cell.getCorner(corner)];
    return slope / voxel;
}

float sampleSdf(Arguments arguments, float3 position) {
    return sampleScalar(arguments.sdf, getCell(arguments.grid, position));
}

float3 sampleSdfSlope(Arguments arguments, float3 position) {
    return sampleSlope(arguments.sdf, getCell(arguments.grid, position), arguments.grid.origin.w);
}

void addScalarGradient(AtomicPtr<float> gradients, Cell cell, float gradient) {
    if (gradient == 0.0 || !isfinite(gradient))
        return;
    [ForceUnroll]
    for (uint corner = 0; corner < 8; ++corner)
        atomicAdd(gradients + cell.getCorner(corner), cell.getWeight(corner) * gradient);
}

void getShBasis(float3 direction, uint active, out float basis[shCoefficients]) {
    const float x = direction.x;
    const float y = direction.y;
    const float z = direction.z;
    basis[0] = 0.28209479177387814;
    basis[1] = -0.4886025119029199 * y;
    basis[2] = 0.4886025119029199 * z;
    basis[3] = -0.4886025119029199 * x;
    basis[4] = 1.0925484305920792 * x * y;
    basis[5] = -1.0925484305920792 * y * z;
    basis[6] = 0.31539156525252005 * (2.0 * z * z - x * x - y * y);
    basis[7] = -1.0925484305920792 * x * z;
    basis[8] = 0.5462742152960396 * (x * x - y * y);
    for (uint coefficient = active; coefficient < shCoefficients; ++coefficient)
        basis[coefficient] = 0.0;
}

float3 sigmoid(float3 value) {
    return 1.0 / (1.0 + exp(-value));
}

float3 getCornerLogits(Arguments arguments, uint node, float basis[shCoefficients]) {
    const Ptr<float> values = arguments.colour + node * colourChannels;
    float3 logits = float3(0.0);
    for (uint coefficient = 0; coefficient < arguments.activeCoefficients; ++coefficient)
        logits += basis[coefficient] * float3(values[coefficient * 3], values[coefficient * 3 + 1], values[coefficient * 3 + 2]);
    return logits;
}

float3 getColourLogits(Arguments arguments, Cell cell, float basis[shCoefficients]) {
    float3 logits = float3(0.0);
    [ForceUnroll]
    for (uint corner = 0; corner < 8; ++corner)
        logits += cell.getWeight(corner) * getCornerLogits(arguments, cell.getCorner(corner), basis);
    return logits;
}

//! The derivative of the colour logits along the direction, in world units.
float3 getColourLogitSlope(Arguments arguments, Cell cell, float basis[shCoefficients], float3 direction) {
    float3 slope = float3(0.0);
    [ForceUnroll]
    for (uint corner = 0; corner < 8; ++corner)
        slope += dot(cell.getSlope(corner), direction) * getCornerLogits(arguments, cell.getCorner(corner), basis);
    return slope / arguments.grid.origin.w;
}

void addColourGradient(Arguments arguments, Cell cell, float basis[shCoefficients], float3 logitGradient) {
    [ForceUnroll]
    for (uint corner = 0; corner < 8; ++corner) {
        const float3 gradient = cell.getWeight(corner) * logitGradient;
        const AtomicPtr<float> gradients = arguments.colourGradient + cell.getCorner(corner) * colourChannels;
        for (uint coefficient = 0; coefficient < arguments.activeCoefficients; ++coefficient) {
            atomicAdd(gradients + coefficient * 3, basis[coefficient] * gradient.x);
            atomicAdd(gradients + coefficient * 3 + 1, basis[coefficient] * gradient.y);
            atomicAdd(gradients + coefficient * 3 + 2, basis[coefficient] * gradient.z);
        }
    }
}

float2 getEnvironmentCoordinate(Arguments arguments, float3 direction) {
    const float u = (atan2(direction.y, direction.x) / (2.0 * pi) + 0.5) * float(arguments.environmentWidth) - 0.5;
    const float v = acos(clamp(direction.z, -1.0, 1.0)) / pi * float(arguments.environmentHeight) - 0.5;
    return float2(u, clamp(v, 0.0, float(arguments.environmentHeight - 1)));
}

uint getEnvironmentTexel(Arguments arguments, int2 texel) {
    const uint x = uint((texel.x % int(arguments.environmentWidth) + int(arguments.environmentWidth)) % int(arguments.environmentWidth));
    const uint y = uint(clamp(texel.y, 0, int(arguments.environmentHeight) - 1));
    return (y * arguments.environmentWidth + x) * 3;
}

float3 getEnvironmentLogits(Arguments arguments, float3 direction, out int2 base, out float2 fraction) {
    const float2 coordinate = getEnvironmentCoordinate(arguments, direction);
    base = int2(floor(coordinate));
    fraction = coordinate - float2(base);
    float3 logits = float3(0.0);
    [ForceUnroll]
    for (uint corner = 0; corner < 4; ++corner) {
        const int2 offset = int2(int(corner & 1), int(corner >> 1));
        const float2 weights = lerp(1.0 - fraction, fraction, float2(offset));
        const uint texel = getEnvironmentTexel(arguments, base + offset);
        logits += weights.x * weights.y * float3(arguments.environment[texel], arguments.environment[texel + 1], arguments.environment[texel + 2]);
    }
    return logits;
}

float3 getBackground(Arguments arguments, float3 direction) {
    if (arguments.background.w == 0.0)
        return arguments.background.xyz;
    int2 base;
    float2 fraction;
    return sigmoid(getEnvironmentLogits(arguments, direction, base, fraction));
}

void addBackgroundGradient(Arguments arguments, float3 direction, float3 colourGradient) {
    if (arguments.background.w == 0.0)
        return;
    int2 base;
    float2 fraction;
    const float3 colour = sigmoid(getEnvironmentLogits(arguments, direction, base, fraction));
    const float3 logitGradient = colourGradient * colour * (1.0 - colour);
    [ForceUnroll]
    for (uint corner = 0; corner < 4; ++corner) {
        const int2 offset = int2(int(corner & 1), int(corner >> 1));
        const float2 weights = lerp(1.0 - fraction, fraction, float2(offset));
        const uint texel = getEnvironmentTexel(arguments, base + offset);
        const float3 gradient = weights.x * weights.y * logitGradient;
        atomicAdd(arguments.environmentGradient + texel, gradient.x);
        atomicAdd(arguments.environmentGradient + texel + 1, gradient.y);
        atomicAdd(arguments.environmentGradient + texel + 2, gradient.z);
    }
}
