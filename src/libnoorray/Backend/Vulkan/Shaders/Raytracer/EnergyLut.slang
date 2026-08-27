#pragma once

// A 1:1 port of Materials/Shading/EnergyLut.h.
//
// Every table lives in one unorm16 storage buffer and uses the manual-lerp
// path, matching the CPU reference so results are bit-identical.

static const int GgxRoughnessSize = 32;
static const int GgxCosineSize = 32;
static const int DielectricF0Size = 16;
static const int DielectricRoughnessSize = 16;
static const int DielectricCosineSize = 16;
static const int GlassIorSize = 32;
static const int GlassIorHalfSize = GlassIorSize / 2;
static const int GlassRoughnessSize = 16;
static const int GlassCosineSize = 16;
static const float MaximumDielectricF0 = 0.08;
static const float MaximumGlassIor = 3.5;
static const float MaximumGlassZ = 0.7453559924999299;

// Element offsets into the packed table, in uint16 units. Mirrors
// VulkanEnergyLutTables on the host side.
static const uint EnergyLutGgxDirectionalOffset = 0u;
static const uint EnergyLutGgxAverageOffset =
    EnergyLutGgxDirectionalOffset + uint(GgxCosineSize * GgxRoughnessSize);
static const uint EnergyLutDielectricDirectionalOffset =
    EnergyLutGgxAverageOffset + uint(GgxRoughnessSize);
static const uint EnergyLutDielectricAverageOffset =
    EnergyLutDielectricDirectionalOffset
    + uint(DielectricCosineSize * DielectricRoughnessSize * DielectricF0Size);
static const uint EnergyLutGlassDirectionalOffset =
    EnergyLutDielectricAverageOffset
    + uint(DielectricRoughnessSize * DielectricF0Size);
static const uint EnergyLutGlassAverageOffset =
    EnergyLutGlassDirectionalOffset
    + uint(GlassCosineSize * GlassRoughnessSize * GlassIorSize);
static const uint EnergyLutElementCount =
    EnergyLutGlassAverageOffset + uint(GlassRoughnessSize * GlassIorSize);

float clampUnit(float value)
{
    return clamp(value, 0.0, 1.0);
}

// Tables are unorm16, two elements per 32-bit word.
float energyLutDecode(ByteAddressBuffer tables, uint element)
{
    const uint word = tables.Load((element >> 1u) << 2u);
    const uint value = (element & 1u) != 0u ? (word >> 16u) : (word & 0xFFFFu);
    return float(value) * (1.0 / 65535.0);
}

float energyLutSample2d(ByteAddressBuffer tables, uint base, int width, int height,
    float x, float y)
{
    const float px = clampUnit(x) * float(width - 1);
    const float py = clampUnit(y) * float(height - 1);
    const int x0 = int(floor(px));
    const int y0 = int(floor(py));
    const int x1 = x0 < width - 1 ? x0 + 1 : x0;
    const int y1 = y0 < height - 1 ? y0 + 1 : y0;
    const float tx = px - float(x0);
    const float ty = py - float(y0);
    return lerp(
        lerp(energyLutDecode(tables, base + uint(y0 * width + x0)),
             energyLutDecode(tables, base + uint(y0 * width + x1)), tx),
        lerp(energyLutDecode(tables, base + uint(y1 * width + x0)),
             energyLutDecode(tables, base + uint(y1 * width + x1)), tx), ty);
}

float energyLutSample3d(ByteAddressBuffer tables, uint base, int width, int height,
    int depth, float x, float y, float z)
{
    const float pz = clampUnit(z) * float(depth - 1);
    const int z0 = int(floor(pz));
    const int z1 = z0 < depth - 1 ? z0 + 1 : z0;
    const float tz = pz - float(z0);
    const uint plane = uint(width * height);
    return lerp(
        energyLutSample2d(tables, base + uint(z0) * plane, width, height, x, y),
        energyLutSample2d(tables, base + uint(z1) * plane, width, height, x, y), tz);
}

float cosineCoordinate(float cosine)
{
    return sqrt(clampUnit(cosine));
}

float dielectricF0Coordinate(float f0)
{
    return sqrt(clampUnit(f0 / MaximumDielectricF0));
}

float glassIorCoordinate(float relativeIor)
{
    const float eta = max(relativeIor, 1.0e-5);
    const bool exiting = eta < 1.0;
    const float ior = exiting ? 1.0 / eta : eta;
    const float z = sqrt(max((ior - 1.0) / (ior + 1.0), 0.0));
    const float side = clampUnit(z / MaximumGlassZ);
    const float index = exiting
        ? float(GlassIorHalfSize - 1) * (1.0 - side)
        : float(GlassIorHalfSize) + float(GlassIorHalfSize - 1) * side;
    return index / float(GlassIorSize - 1);
}

float ggxDirectionalAlbedo(ByteAddressBuffer tables, float roughness, float cosine)
{
    return energyLutSample2d(tables, EnergyLutGgxDirectionalOffset,
        GgxCosineSize, GgxRoughnessSize,
        cosineCoordinate(cosine), clampUnit(roughness));
}

float ggxAverageAlbedo(ByteAddressBuffer tables, float roughness)
{
    return energyLutSample2d(tables, EnergyLutGgxAverageOffset,
        GgxRoughnessSize, 1, clampUnit(roughness), 0.0);
}

float dielectricDirectionalAlbedo(ByteAddressBuffer tables, float roughness,
    float cosine, float normalReflectance)
{
    return energyLutSample3d(tables, EnergyLutDielectricDirectionalOffset,
        DielectricCosineSize, DielectricRoughnessSize, DielectricF0Size,
        cosineCoordinate(cosine), clampUnit(roughness),
        dielectricF0Coordinate(normalReflectance));
}

float dielectricAverageAlbedo(ByteAddressBuffer tables, float roughness,
    float normalReflectance)
{
    return energyLutSample2d(tables, EnergyLutDielectricAverageOffset,
        DielectricRoughnessSize, DielectricF0Size,
        clampUnit(roughness), dielectricF0Coordinate(normalReflectance));
}

float glassDirectionalAlbedo(ByteAddressBuffer tables, float roughness,
    float cosine, float relativeIor)
{
    if (abs(relativeIor - 1.0) < 1.0e-4)
        return 1.0;
    return energyLutSample3d(tables, EnergyLutGlassDirectionalOffset,
        GlassCosineSize, GlassRoughnessSize, GlassIorSize,
        cosineCoordinate(cosine), clampUnit(roughness),
        glassIorCoordinate(relativeIor));
}

float glassAverageAlbedo(ByteAddressBuffer tables, float roughness, float relativeIor)
{
    if (abs(relativeIor - 1.0) < 1.0e-4)
        return 1.0;
    return energyLutSample2d(tables, EnergyLutGlassAverageOffset,
        GlassRoughnessSize, GlassIorSize,
        clampUnit(roughness), glassIorCoordinate(relativeIor));
}

// Kulla-Conty multi-scatter compensation: the energy a single-scatter GGX
// evaluation misses at high roughness, redistributed as an isotropic term.
float multipleScatterFresnel(float averageFresnelValue, float averageAlbedo)
{
    const float fresnel = clampUnit(averageFresnelValue);
    const float albedo = clampUnit(averageAlbedo);
    return fresnel * fresnel * albedo / max(1.0 - fresnel * (1.0 - albedo), 1.0e-6);
}
