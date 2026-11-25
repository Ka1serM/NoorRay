#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "NoorRay/Core/Error.h"
#include "NoorRay/Core/Math.h"

namespace NoorRay::Materials {

//! Operations of a material graph. Element-wise operations take operands of the result width or of width 1, which broadcast.
enum class GraphOp : std::uint8_t {
    Constant,
    Input,
    Parameter,
    Texture,
    Call,
    Extract,
    Combine,
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
    Atan2,
    Modulo,
    FlooredModulo,
    Abs,
    Floor,
    Ceil,
    Fract,
    Round,
    Trunc,
    Sign,
    Sqrt,
    InverseSqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Less,
    Greater,
    Select,
    Dot,
    Cross,
    Length,
    Distance,
    Normalize,
    Reflect,
    Refract,
    FaceForward,
    Clamp,
    Mix,
    Smoothstep,
    MulAdd,
};
inline constexpr std::size_t graphOpCount = static_cast<std::size_t>(GraphOp::MulAdd) + 1;

//! Values of the hit that a graph reads; graphInputLayout gives where each one lives in Shaders::ShadingContext.
enum class GraphInput : std::uint8_t {
    Position,
    Normal,
    GeometricNormal,
    Tangent,
    Incoming,
    Uv0,
    Uv1,
    Uv2,
    Uv3,
    Color,
    ObjectPosition,
    Generated,
    Backfacing,
    Distance,
    ObjectRandom,
    ClosureRandom,
    DPdx,
    DPdy,
    UvGradients,
    ObjectLocation,
    Time,
    ObjectDx,
    ObjectDy,
    GeneratedScale,
    WorldToObject0,
    WorldToObject1,
    WorldToObject2,
    ObjectToWorld0,
    ObjectToWorld1,
    ObjectToWorld2,
    WorldToCamera0,
    WorldToCamera1,
    WorldToCamera2,
    CameraWindow,
    IsCameraRay,
    IsShadowRay,
    IsDiffuseRay,
    IsGlossyRay,
    IsSingularRay,
    IsReflectionRay,
    IsTransmissionRay,
    IsVolumeScatterRay,
    RayDepth,
    DiffuseDepth,
    GlossyDepth,
    TransmissionDepth,
    ObjectColor,
    ObjectIndex,
    TransparentDepth,
    CameraPerspective,
};
inline constexpr std::size_t graphInputCount = static_cast<std::size_t>(GraphInput::CameraPerspective) + 1;

struct GraphInputLayout {
    std::uint8_t member;
    std::uint8_t component;
    std::uint8_t width;
};

//! The members of Shaders::ShadingContext, all float4, that a graph reads.
inline constexpr std::size_t graphInputMembers = 32;
GraphInputLayout graphInputLayout(GraphInput input);

//! Host functions in Shaders/Materials/Graph.slang that a graph calls by name; Parameter and Texture take the material data first, the rest float vectors.
enum class GraphFunction : std::uint8_t {
    Parameter,
    Texture,
    Noise,
    Voronoi,
    VoronoiPosition,
    Wave,
    Magic,
    Checker,
    Brick,
    Gradient,
    WhiteNoise,
    HueSaturationValue,
    RgbToHsv,
    HsvToRgb,
    Blackbody,
    Wavelength,
    Fresnel,
    LayerWeight,
    AmbientOcclusion,
    Bevel,
    Raycast,
    RaycastNormal,
    Gabor,
    RadialTiling,
    RadialTilingSegment,
    BoxWeights,
    InstanceAttribute,
    Parametric,
    Wireframe,
};
inline constexpr std::size_t graphFunctionCount = static_cast<std::size_t>(GraphFunction::Wireframe) + 1;

struct GraphFunctionInfo {
    std::string_view name;
    //! The width of each parameter; the list ends at the first zero.
    std::array<std::uint8_t, 6> parameters;
    std::uint8_t result;

    std::size_t parameterCount() const;
};

const GraphFunctionInfo& graphFunctionInfo(GraphFunction function);

//! Each output is a member of Shaders::MaterialGraphOutput, a float4 but for the transparency; the first thirteen are the members of Shaders::BsdfInputs.
enum class GraphOutput : std::uint8_t {
    BaseColor,
    MetallicRoughnessIorSpecular,
    SpecularTintAnisotropic,
    NormalRotation,
    TangentTransmission,
    Coat,
    CoatTint,
    CoatNormal,
    Sheen,
    SheenTint,
    Emission,
    ThinFilmDispersion,
    //! The weight of the chosen closure divided by its probability, and w the kind of the closure: 1 translucent, 2 holdout, 3 refraction without reflection.
    Closure,
    //! xyz is the subsurface radius in Blender units, w the subsurface weight.
    Subsurface,
    //! x is the transparency; the member is a float.
    Transparency,
};
inline constexpr std::size_t graphOutputCount = static_cast<std::size_t>(GraphOutput::Transparency) + 1;

//! index is the constant, input, parameter slot, texture, function or component; the operands are operands[first, first + count).
struct GraphInstruction {
    GraphOp op = GraphOp::Constant;
    std::uint8_t width = 1;
    std::uint16_t index = 0;
    std::uint32_t first = 0;
    std::uint32_t count = 0;

    bool operator==(const GraphInstruction&) const = default;
};

//! An editable value: a float4 slot of the material parameters, of which the graph reads the first width components.
struct GraphParameter {
    std::string name;
    std::uint32_t slot = 0;
    std::uint8_t width = 1;

    bool operator==(const GraphParameter&) const = default;
};

//! A flat SSA program of float vectors: the value of instruction n is value n. Materials share a graph when their programs are equal.
class Graph {
public:
    struct Source {
        std::vector<GraphInstruction> instructions;
        std::vector<std::uint32_t> operands;
        std::vector<float> constants;
        std::vector<GraphParameter> parameters;
        std::uint32_t parameterSlots = 0;
        std::uint32_t textureCount = 0;
        std::array<std::uint32_t, graphOutputCount> outputs{};
    };

    [[nodiscard]] std::expected<void, Core::Error> open(Source source, std::string_view materialName);

    const Source& getSource() const { return source; }
    std::span<const GraphInstruction> getInstructions() const { return source.instructions; }
    std::span<const std::uint32_t> operands(const GraphInstruction& instruction) const {
        return std::span(source.operands).subspan(instruction.first, instruction.count);
    }
    std::span<const float> getConstants() const { return source.constants; }
    std::span<const GraphParameter> getParameters() const { return source.parameters; }
    std::uint32_t getParameterSlots() const { return source.parameterSlots; }
    std::uint32_t getTextureCount() const { return source.textureCount; }
    std::uint32_t getOutput(const GraphOutput output) const { return source.outputs[static_cast<std::size_t>(output)]; }
    //! The constant value of a value, when it is a constant.
    const float* constantOf(std::uint32_t value) const;
    //! The value from the constants and parameter values alone, or none when it reads the hit, a texture or a function.
    std::optional<Core::float4> evaluateUniform(std::uint32_t value, std::span<const Core::float4> parameters) const;
    std::uint64_t getHash() const { return hash; }
    //! Equal programs; parameter names and values may differ.
    bool hasSameProgram(const Graph& other) const;
    //! Whether the value of output depends on input.
    bool reads(GraphOutput output, GraphInput input) const;

private:
    Source source;
    std::uint64_t hash = 0;
};

//! Builds a graph instruction by instruction and folds operations on constants.
class GraphBuilder {
public:
    std::uint32_t constant(float value);
    std::uint32_t constant(Core::float2 value);
    std::uint32_t constant(Core::float3 value);
    std::uint32_t constant(Core::float4 value);
    std::uint32_t input(GraphInput input);
    std::uint32_t parameter(std::string name, std::uint8_t width);
    //! gradients are the changes of uv across a pixel, along x in xy and along y in zw.
    std::uint32_t texture(std::uint32_t texture, std::uint32_t uv, std::uint32_t gradients);
    std::uint32_t call(GraphFunction function, std::span<const std::uint32_t> operands);
    std::uint32_t extract(std::uint32_t value, std::uint32_t component);
    std::uint32_t combine(std::span<const std::uint32_t> values);
    //! Widens or narrows value to width: a scalar broadcasts, extra components drop, and missing ones are fill.
    std::uint32_t resize(std::uint32_t value, std::uint8_t width, float fill = 0.0f);
    std::uint32_t op(GraphOp op, std::span<const std::uint32_t> operands);
    std::uint32_t op(GraphOp op, std::uint32_t a) { return this->op(op, std::array{a}); }
    std::uint32_t op(GraphOp op, std::uint32_t a, std::uint32_t b) { return this->op(op, std::array{a, b}); }
    std::uint32_t op(GraphOp op, std::uint32_t a, std::uint32_t b, std::uint32_t c) { return this->op(op, std::array{a, b, c}); }

    std::uint8_t width(const std::uint32_t value) const { return source.instructions[value].width; }
    //! The constant value of a value, when it is a constant.
    const float* constantOf(std::uint32_t value) const;
    std::uint32_t addTexture() { return source.textureCount++; }
    //! The values of the parameters, one float4 per slot.
    std::vector<Core::float4>& parameterValues() { return values; }
    void setOutput(GraphOutput output, std::uint32_t value);
    Graph::Source take() { return std::move(source); }

private:
    std::uint32_t emit(GraphInstruction instruction, std::span<const std::uint32_t> operands);
    std::uint32_t constantValues(std::span<const float> values);
    std::optional<std::uint32_t> wholeOfExtracts(std::span<const std::uint32_t> values) const;
    std::uint8_t resultWidth(GraphOp op, std::span<const std::uint32_t> operands) const;
    std::uint32_t fold(GraphOp op, std::uint8_t width, std::span<const std::uint32_t> operands);

    Graph::Source source;
    std::vector<Core::float4> values;
    //! The value of each instruction by its op, width, index and operands, or by its components for a constant.
    std::unordered_map<std::u32string, std::uint32_t> known;
};

}
