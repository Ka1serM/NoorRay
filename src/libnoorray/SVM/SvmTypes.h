#pragma once

#include <cstdint>

#include "CUDA/Annotations.h"

// Shader Virtual Machine bytecode format, ported from Blender Cycles'
// kernel/svm/types.h + scene/svm.h (intern/cycles). Cycles compiles its own
// ShaderGraph to this format; NoorRay's SvmCompiler (SvmCompiler.h) instead
// compiles a MaterialX::DocumentPtr to it directly, so every opcode below
// corresponds to a MaterialX node category rather than a Cycles ShaderNode
// subclass. Where MaterialX has no Cycles equivalent (open_pbr_surface,
// artistic_ior, ...) new opcodes are added that target NoorRayCompositeBsdf's
// lobe kinds (Shading/CompositeBsdf.h) directly instead of Cycles' own
// principled-BSDF split.
namespace nr::svm
{

struct SvmProgramRecord
{
    std::uint32_t wordOffset{};
    std::uint32_t wordCount{};
    std::uint32_t textureOffset{};
    std::uint32_t textureCount{};
};

using StackOffset = std::uint8_t;

// SVM stack has a fixed size; offsets in [0, StackSize). InvalidOffset marks
// "not on the stack" (e.g. an unused/optional input).
inline constexpr int StackSize = 255;
inline constexpr StackOffset InvalidOffset = 255;

// A node float input is either a literal float value or a stack offset,
// distinguished by encoding the offset as a quiet-NaN bit pattern (mirrors
// Cycles' SVM_INPUT_STACK_OFFSET_MASK) so both fit in one 32-bit word with
// no separate is-linked flag. This is what SvmCompiler::inputFloat() /
// SvmEval's stackOrLiteral() below share.
inline constexpr std::uint32_t StackOffsetNanMask = 0x7FC00000u;

NR_CPU_GPU inline std::uint32_t encodeStackOffset(StackOffset offset)
{
    return StackOffsetNanMask | static_cast<std::uint32_t>(offset);
}

// True if `bits` (as produced by encodeStackOffset) refers to a stack slot
// rather than being a literal float's raw bit pattern. A real finite float
// can never collide with this pattern -- only NaNs have every exponent bit
// set, and this mask additionally pins the two top mantissa bits, keeping
// the encoding clear of ordinary NaNs a graph might legitimately produce.
NR_CPU_GPU inline bool isStackOffset(std::uint32_t bits)
{
    return (bits & StackOffsetNanMask) == StackOffsetNanMask;
}

NR_CPU_GPU inline StackOffset decodeStackOffset(std::uint32_t bits)
{
    return static_cast<StackOffset>(bits & 0xFFu);
}

// Opcode list. Phase A covers every node category materialx_export.py
// (the Blender addon's MaterialX exporter) currently emits; Phase B extends
// this with the remaining Cycles kernel/svm/*.h node categories as MaterialX
// coverage grows (see the plan's Phase B backlog: brick/magic/ies/sky/ao/
// bevel/wireframe/camera/light_path/aov/displace/ramp).
enum class NodeType : std::uint32_t
{
    End = 0,

    // Values / conversion
    Value,          // literal float/color3/vector3 already folded into the
                     // consuming instruction's operands -- rarely emitted as
                     // its own node, kept for constant-folding fallback.
    Convert,        // scalar <-> color3 <-> vector3 widen/narrow + luminance
    Math,           // ShaderNodeMath's full op set (add, mul, pow, ...)
    VectorMath,     // ShaderNodeVectorMath's op set (add, cross, dot, ...)
    Mix,            // mix(a, b, t) over float/color3/vector3
    Clamp,
    RemapRange,     // MaterialX <remap>/<smoothstep> per interpolation mode
    Range,          // MaterialX <range>: remap, signed gamma, optional clamp
    Gamma,
    HsvAdjust,
    Invert,
    Contrast,
    Saturate,
    Blackbody,
    SeparateColor,  // extract a channel from color3/vector3
    CombineColor,   // combine3

    // Geometry / textures
    TexCoord,
    Mapping,
    Rotate3d,
    ImageTexture,
    NoiseTexture,
    FractalNoiseTexture,
    CheckerTexture,
    GradientTexture,
    WorleyNoiseTexture,
    CellNoiseTexture,
    UnifiedNoiseTexture,

    // Shading (bump/normal)
    NormalMap,
    Bump,

    // Closures -- each allocates directly into NoorRayCompositeBsdf rather
    // than a Cycles-style ShaderClosure[] array (see SvmEval.h).
    ClosureWeight,      // sets the implicit "tint" register consumed by the
                        // following ClosureBsdf/ClosureEdf instruction
    ClosureDiffuseBsdf,
    ClosureConductorBsdf,   // MaterialX conductor_bsdf / Cycles glossy
    ClosureDielectricBsdf,  // MaterialX dielectric_bsdf (reflection+
                            // transmission, covers glass/refraction/
                            // transparent/coat depending on params)
    ClosureSheenBsdf,       // approximated onto Conductor (see SvmEval.h)
    ClosureSubsurfaceBsdf,  // approximated onto Diffuse (see SvmEval.h)
    ClosureUniformEdf,      // emission
    ClosureOpenPbrSurface,  // MaterialX open_pbr_surface: expands to the
                            // full diffuse+coat+metal+glass lobe set in one
                            // instruction instead of a leaf-BSDF subgraph
    ArtisticIor,            // reflectivity/edge_color -> conductor IOR
    MixClosure,             // binary mix of two already-emitted closures
    AddClosure,             // binary add (unweighted) of two closures
    JumpIfZero,             // multi-closure tree pruning (mirrors Cycles'
    JumpIfOne,              // NODE_JUMP_IF_ZERO/ONE)

    SurfaceOutput,  // terminal instruction: material's final surface result
};

// Packed per-node parameter structs. Every struct here is POD and its size
// must be a multiple of sizeof(uint32_t) -- SvmCompiler::addNode<T>() blits
// it word-by-word after the opcode, matching Cycles' add_node<T>() (scene/
// svm.h). Vector-valued inputs pack as three SVMInputFloat-style words (one
// per component, each independently literal-or-stack-offset) rather than a
// single "is this linked" flag per input, mirroring SVMInputFloat3.

// One enumerant per MaterialX scalar-math node category (each op is its own
// MaterialX node, unlike Cycles' single ShaderNodeMath with an operation
// enum) -- NodeMath::mathType is one of these, so the interpreter still only
// needs one opcode+switch for the whole family.
enum class MathOp : std::uint32_t
{
    Add, Subtract, Multiply, Divide, Modulo, Power, Absval, Floor, Ceil,
    Round, Sign, Min, Max, Sin, Cos, Tan, Asin, Acos, Atan2, Ln, Exp, Sqrt,
    InverseSqrt, Fract, GreaterThan, GreaterEqual, Equal,
    // MaterialX <and>/<or>/<not> (ND_logical_*): boolean inputs/outputs are
    // just 0.0/1.0 floats on the SVM stack, so these reuse NodeMath rather
    // than a dedicated boolean instruction.
    And, Or, Not,
};

enum class VectorMathOp : std::uint32_t
{
    Add, Subtract, Multiply, Divide, CrossProduct, DotProduct, Normalize,
    Magnitude, Distance,
    // Same shared operation family as Cycles' NODE_VECTOR_MATH_REFLECT and
    // NODE_VECTOR_MATH_REFRACT. MaterialX functional graphs map to one SVM
    // instruction rather than expanded scalar/vector nodes.
    Reflect, Refract,
};

// Direct port of Cycles' NodeMix enumeration. MaterialX blend categories map
// to these modes, keeping a complete blend node to one bytecode instruction.
enum class MixBlendType : std::uint32_t
{
    Blend, Add, Multiply, Subtract, Screen, Divide, Difference, Darken,
    Lighten, Overlay, Dodge, Burn, Hue, Saturation, Value, Color, SoftLight,
    LinearLight, Exclusion, Clamp,
};

struct NodeMath
{
    std::uint32_t mathType;
    std::uint32_t value1, value2, value3;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeVectorMath
{
    std::uint32_t mathType;
    std::uint32_t value1X, value1Y, value1Z;
    std::uint32_t value2X, value2Y, value2Z;
    std::uint32_t value3X, value3Y, value3Z;
    StackOffset resultOffset;
    StackOffset resultIsScalar; // dot/length/distance produce a scalar
    StackOffset pad[2]{};
};

struct NodeMix
{
    std::uint32_t fac;
    std::uint32_t value1X, value1Y, value1Z;
    std::uint32_t value2X, value2Y, value2Z;
    StackOffset resultOffset;
    StackOffset isColor; // false: operate on X only (float mix)
    StackOffset pad[2]{};
    MixBlendType blendType{MixBlendType::Blend};
};

struct NodeClamp
{
    std::uint32_t value, minValue, maxValue;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeRemapRange
{
    std::uint32_t value, inLow, inHigh, outLow, outHigh;
    StackOffset resultOffset;
    StackOffset smoothstep;
    StackOffset pad[2]{};
};

struct NodeRange
{
    std::uint32_t valueX, valueY, valueZ;
    std::uint32_t inLowX, inLowY, inLowZ;
    std::uint32_t inHighX, inHighY, inHighZ;
    std::uint32_t gammaX, gammaY, gammaZ;
    std::uint32_t outLowX, outLowY, outLowZ;
    std::uint32_t outHighX, outHighY, outHighZ;
    std::uint32_t doClamp;
    StackOffset resultOffset;
    StackOffset isColor;
    StackOffset pad{};
};

struct NodeGamma
{
    std::uint32_t value, gamma;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeHsvAdjust
{
    std::uint32_t hue, saturation, value, fac;
    std::uint32_t colorX, colorY, colorZ;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeInvert
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t amountX, amountY, amountZ;
    StackOffset resultOffset;
    StackOffset isColor;
    StackOffset pad[2]{};
};

// MaterialX adjustment-library nodes.  Each replaces the corresponding
// standard-library graph (three math nodes for contrast; luminance+mix for
// saturate) with one fixed interpreter instruction.
struct NodeContrast
{
    std::uint32_t valueX, valueY, valueZ;
    std::uint32_t amountX, amountY, amountZ;
    std::uint32_t pivotX, pivotY, pivotZ;
    StackOffset resultOffset;
    StackOffset isColor;
    StackOffset pad[2]{};
};

struct NodeSaturate
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t amount;
    std::uint32_t lumaX, lumaY, lumaZ;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeBlackbody
{
    std::uint32_t temperature;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

enum class ColorChannelLayout : std::uint32_t
{
    Rgb,
    Hsv,
    Hsl,
};

struct NodeSeparateColor
{
    std::uint32_t colorX, colorY, colorZ;
    ColorChannelLayout layout;
    StackOffset resultXOffset, resultYOffset, resultZOffset;
    StackOffset pad{};
};

struct NodeCombineColor
{
    std::uint32_t x, y, z;
    ColorChannelLayout layout;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

enum class TexCoordSource : std::uint32_t
{
    UV,
    Object,
    World,
    Normal,
    Tangent,
    Bitangent,
    VertexColor,
    ViewDirection,
};

struct NodeTexCoord
{
    TexCoordSource source;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeMapping
{
    std::uint32_t inX, inY, inZ;
    std::uint32_t translationX, translationY, translationZ;
    std::uint32_t rotationX, rotationY, rotationZ; // radians
    std::uint32_t scaleX, scaleY, scaleZ;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeRotate3d
{
    std::uint32_t inX, inY, inZ;
    std::uint32_t amountDegrees;
    std::uint32_t axisX, axisY, axisZ;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeImageTexture
{
    std::int32_t textureSlot; // index into SvmCompiledProgram::textureIndices
    std::uint32_t uvX, uvY;
    StackOffset resultColorOffset;
    StackOffset resultAlphaOffset; // InvalidOffset if alpha unused
    StackOffset uAddressMode; // 0 periodic, 1 clamp, 2 mirror, 3 constant
    StackOffset vAddressMode;
    StackOffset filterType;   // 0 linear/cubic, 1 closest
    StackOffset pad{};
};

struct NodeProceduralTexture
{
    std::uint32_t posX, posY, posZ;
    std::uint32_t amplitudeX, amplitudeY, amplitudeZ;
    // These are intentionally encoded as ordinary SVM inputs.  MaterialX's
    // fractal3d node takes a position, integer octaves, lacunarity and
    // diminish; keeping each input dynamic matches the standard node's
    // semantics when any port is connected.
    std::uint32_t octaves;
    std::uint32_t lacunarity;
    std::uint32_t diminish;
    StackOffset resultOffset;
    StackOffset resultIsVector;
    StackOffset is2d;
    StackOffset pad{};
};

struct NodeNoiseTexture
{
    std::uint32_t posX, posY, posZ;
    std::uint32_t amplitudeX, amplitudeY, amplitudeZ;
    std::uint32_t pivot;
    StackOffset resultOffset;
    StackOffset resultIsVector;
    StackOffset is2d;
    StackOffset pad{};
};

struct NodeWorleyNoiseTexture
{
    std::uint32_t posX, posY, posZ;
    std::uint32_t jitter;
    std::uint32_t style;
    StackOffset resultOffset;
    StackOffset resultIsVector;
    StackOffset is2d;
    StackOffset pad{};
};

struct NodeCellNoiseTexture
{
    std::uint32_t posX, posY, posZ;
    StackOffset resultOffset;
    StackOffset is2d;
    StackOffset pad[2]{};
};

// Direct form of stdlib NG_unifiednoise2d/3d_float.  Keeping its semantic
// switch in one instruction avoids emitting the reference nodegraph's
// coordinate, rotate, noise, switch and range temporaries per material.
struct NodeUnifiedNoiseTexture
{
    std::uint32_t posX, posY, posZ;
    std::uint32_t freqX, freqY, freqZ;
    std::uint32_t offsetX, offsetY, offsetZ;
    std::uint32_t jitter;
    std::uint32_t outMin, outMax;
    std::uint32_t clampOutput;
    std::uint32_t octaves, lacunarity, diminish;
    std::uint32_t noiseType, style;
    StackOffset resultOffset;
    StackOffset is2d;
    StackOffset pad[2]{};
};

struct NodeCheckerTexture
{
    std::uint32_t color1X, color1Y, color1Z;
    std::uint32_t color2X, color2Y, color2Z;
    std::uint32_t tilingX, tilingY;
    std::uint32_t offsetX, offsetY;
    std::uint32_t uvX, uvY;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeNormalMap
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t scaleX, scaleY;
    std::uint32_t normalX, normalY, normalZ;
    std::uint32_t tangentX, tangentY, tangentZ;
    std::uint32_t bitangentX, bitangentY, bitangentZ;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeBump
{
    std::uint32_t height, strength, distance;
    StackOffset resultOffset;
    StackOffset pad[3]{};
};

struct NodeClosureWeight
{
    std::uint32_t weightX, weightY, weightZ;
};

struct NodeClosureDiffuseBsdf
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t roughness;
    std::uint32_t normalX, normalY, normalZ; // InvalidOffset-encoded => use shading normal
    std::uint32_t burley;
    std::uint32_t translucent;
};

struct NodeClosureConductorBsdf
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t roughness;
    std::uint32_t anisotropy;
    std::uint32_t iorX, iorY, iorZ;
    std::uint32_t extinctionX, extinctionY, extinctionZ;
    std::uint32_t normalX, normalY, normalZ;
};

struct NodeClosureDielectricBsdf
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t roughness;
    std::uint32_t ior;
    std::uint32_t transmission; // 0 = reflection-only (coat/glossy), 1 = full glass
    std::uint32_t normalX, normalY, normalZ;
};

struct NodeClosureSheenBsdf
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t roughness;
    std::uint32_t normalX, normalY, normalZ;
};

struct NodeClosureSubsurfaceBsdf
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t roughness;
    std::uint32_t normalX, normalY, normalZ;
};

struct NodeClosureUniformEdf
{
    std::uint32_t colorX, colorY, colorZ;
    std::uint32_t strength;
};

struct NodeClosureOpenPbrSurface
{
    std::uint32_t baseColorX, baseColorY, baseColorZ;
    std::uint32_t baseWeight, baseDiffuseRoughness;
    std::uint32_t metalness;
    std::uint32_t specularWeight, specularRoughness, specularIor;
    std::uint32_t specularColorX, specularColorY, specularColorZ;
    std::uint32_t transmissionWeight, transmissionColorX, transmissionColorY, transmissionColorZ;
    std::uint32_t subsurfaceWeight, subsurfaceColorX, subsurfaceColorY, subsurfaceColorZ;
    std::uint32_t fuzzWeight, fuzzColorX, fuzzColorY, fuzzColorZ, fuzzRoughness;
    std::uint32_t coatWeight, coatColorX, coatColorY, coatColorZ, coatRoughness, coatIor;
    std::uint32_t normalX, normalY, normalZ;
    std::uint32_t emissionColorX, emissionColorY, emissionColorZ, emissionLuminance;
    std::uint32_t opacity;
};

// MaterialX ND_artistic_ior is a value node with two color3 outputs.  It is
// intentionally a single instruction: the standard-library implementation
// computes both values from the same reflectivity/edge_color pair.
struct NodeArtisticIor
{
    std::uint32_t reflectivityX, reflectivityY, reflectivityZ;
    std::uint32_t edgeTintX, edgeTintY, edgeTintZ;
    std::uint32_t resultIorX, resultIorY, resultIorZ;
    std::uint32_t resultExtinctionX, resultExtinctionY, resultExtinctionZ;
};

struct NodeMixClosure
{
    std::uint32_t fac;
};

// NODE_JUMP_IF_ZERO/ONE payload: `offset` is how many instruction words to
// skip (relative, from just after this instruction) when the condition
// holds, mirroring Cycles' multi-closure tree pruning in
// generate_multi_closure() (scene/svm.cpp).
struct NodeJump
{
    std::uint32_t condition;
    std::uint32_t offset;
};

} // namespace nr::svm
