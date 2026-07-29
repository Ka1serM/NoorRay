// MaterialX front end for the Cycles-derived SVM stream.
//
// The execution model and bytecode stream are deliberately independent from
// MaterialX: MaterialX is only used here to resolve a typed graph into the
// exact instruction data consumed by SvmEval.  This keeps the device program
// fixed, like Cycles' SVMCompiler + kernel/svm/svm.h pairing.
#include "SVM/SvmCompiler.h"

#include <bit>
#include <cstring>
#include <limits>
#include <array>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>

#include "SVM/SvmTypes.h"

namespace mx = MaterialX;

namespace nr::svm
{
namespace
{
class Emitter
{
public:
    template <typename T>
    void add(const NodeType type, const T& node)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(sizeof(T) % sizeof(std::uint32_t) == 0);
        words_.push_back(static_cast<std::uint32_t>(type));
        const std::size_t wordCount = sizeof(T) / sizeof(std::uint32_t);
        const std::size_t begin = words_.size();
        words_.resize(begin + wordCount);
        std::memcpy(words_.data() + begin, &node, sizeof(T));
    }

    void end() { words_.push_back(static_cast<std::uint32_t>(NodeType::End)); }
    std::size_t size() const { return words_.size(); }
    void patch(const std::size_t index, const std::uint32_t value)
    {
        if (index >= words_.size())
            throw SvmCompileError("SVM: attempted to patch an invalid jump");
        words_[index] = value;
    }
    std::vector<std::uint32_t> take() { return std::move(words_); }

private:
    std::vector<std::uint32_t> words_;
};

std::uint32_t floatWord(const float value)
{
    return std::bit_cast<std::uint32_t>(value);
}

// Categories emitNode()/emitClosure() below dispatch on directly. Anything
// else (UsdPreviewSurface, gltf_pbr, standard_surface_to_*, roughness_dual,
// switch, ramp, tiledimage, ...) is a MaterialX standard-library *nodegraph*
// -- a functional graph built entirely out of these primitives, with no
// dedicated source implementation of its own -- so flattenNodeGraphs() below
// inlines it into its constituent primitive nodes instead of needing a
// hand-written translation here. Keep this set in sync with the category
// strings matched in emitNode()/emitClosure(); a category missing from here
// that also lacks a nodegraph implementation still fails compilation with a
// clear "unsupported" error, same as before this set existed.
bool isNativelySupportedCategory(const std::string& category)
{
    static const std::unordered_set<std::string> categories{
        // Value nodes (emitNode)
        "constant", "dot", "and", "or", "not",
        "add", "subtract", "multiply", "divide", "min", "max", "modulo", "power", "atan2",
        "absval", "floor", "ceil", "round", "sign", "sin", "cos", "tan", "asin", "acos",
        "ln", "exp", "sqrt", "fract",
        "mix", "plus", "minus", "difference", "burn", "dodge", "screen", "overlay",
        "ifgreater", "ifgreatereq", "ifequal",
        "magnitude", "dotproduct", "crossproduct", "distance", "normalize", "reflect", "refract",
        "combine2", "combine3", "separate2", "separate3",
        "remap", "smoothstep", "range",
        "hsvadjust", "invert", "contrast", "saturate", "rgbtohsv", "hsvtorgb", "clamp",
        "texcoord", "position", "normal", "tangent", "bitangent", "geomcolor", "viewdirection",
        "rotate3d", "image", "normalmap", "convert", "luminance", "blackbody", "extract",
        "fractal2d", "fractal3d", "noise2d", "noise3d", "worleynoise2d", "worleynoise3d",
        "cellnoise2d", "cellnoise3d", "unifiednoise2d", "unifiednoise3d",
        "checkerboard", "artistic_ior",
        // Closure nodes (emitClosure)
        "surface", "layer", "add", "mix", "open_pbr_surface", "standard_surface",
        "oren_nayar_diffuse_bsdf", "burley_diffuse_bsdf", "translucent_bsdf",
        "dielectric_bsdf", "conductor_bsdf", "sheen_bsdf", "subsurface_bsdf", "uniform_edf",
        "conical_edf", "generalized_schlick_edf", "generalized_schlick_bsdf",
        // Top-level wrappers SvmCompiler::compile() searches for directly
        "surfacematerial", "volumematerial",
    };
    return categories.contains(category);
}

// Inlines every node whose MaterialX implementation is a standard-library
// <nodegraph> (rather than native source code) into its constituent nodes,
// recursively, using MaterialX's own graph-rewriting utility -- the same
// mechanism a shader generator uses to flatten functional graphs, just
// applied ahead of SvmCompiler's own fixed primitive set instead of GLSL/OSL
// text generation. Nodes already handled by isNativelySupportedCategory are
// left untouched (the filter returns false for them) so previously-compiled
// materials emit identical bytecode.
void flattenNodeGraphs(const mx::DocumentPtr& document)
{
    // NodeDef::getImplementation(target) special-cases an *empty* target to
    // mean "first matching implementation, in library load order" -- for
    // nodes like gltf_pbr that ship both a target-agnostic nodegraph and
    // per-target (genglsl/genosl/genmdl) source-code implementations, that
    // can silently pick the source implementation over the nodegraph.  Pass
    // a target string no library defines a TargetDef for instead: with no
    // candidate targets, getImplementation() falls through to its "generic
    // match" branch, which explicitly means the target-agnostic
    // implementation -- the nodegraph, whenever one exists.
    const mx::NodePredicate filter = [](const mx::NodePtr& node) {
        return !isNativelySupportedCategory(node->getCategory());
    };
    document->flattenSubgraphs("none", filter);
    // flattenSubgraphs() only walks its own direct GraphElement (here, the
    // document's top-level nodes) -- a material's own authored <nodegraph>
    // (for example a hand-built normal-map chain referenced via
    // input/nodegraph="...") is a separate GraphElement whose children it
    // never visits, so flatten each one too.
    for (const mx::NodeGraphPtr& nodeGraph : document->getNodeGraphs())
        nodeGraph->flattenSubgraphs("none", filter);
}

float floatInput(const mx::NodePtr& node, const char* name, const float fallback)
{
    const mx::InputPtr input = node->getInput(name);
    if (!input || input->getValueString().empty())
        return fallback;
    try {
        const mx::ValuePtr value = node->getInputValue(name);
        if (!value)
            return fallback;
        if (input->getType() == "integer")
            return static_cast<float>(value->asA<int>());
        if (input->getType() == "boolean")
            return value->asA<bool>() ? 1.0f : 0.0f;
        return value->asA<float>();
    }
    catch (const mx::Exception&) {
        throw SvmCompileError("SVM: expected float input '" + std::string(name)
            + "' on MaterialX node '" + node->getName() + "'");
    }
}

mx::Color3 colorInput(const mx::NodePtr& node, const char* name,
    const mx::Color3& fallback)
{
    const mx::InputPtr input = node->getInput(name);
    if (!input || input->getValueString().empty())
        return fallback;
    try {
        const mx::ValuePtr value = node->getInputValue(name);
        if (!value)
            return fallback;
        // MaterialX deliberately distinguishes color3 from vector3 even
        // though both occupy three scalar SVM stack slots.  Read the port's
        // authored type rather than assuming every vector-valued input is a
        // color; Wave and Mapping graphs use vector3 throughout.
        if (input->getType() == "vector3") {
            const mx::Vector3 v = value->asA<mx::Vector3>();
            return {v[0], v[1], v[2]};
        }
        if (input->getType() == "vector2") {
            const mx::Vector2 v = value->asA<mx::Vector2>();
            return {v[0], v[1], 0.0f};
        }
        if (input->getType() == "vector4") {
            const mx::Vector4 v = value->asA<mx::Vector4>();
            return {v[0], v[1], v[2]};
        }
        return value->asA<mx::Color3>();
    }
    catch (const mx::Exception&) {
        throw SvmCompileError("SVM: expected color3 input '" + std::string(name)
            + "' on MaterialX node '" + node->getName() + "'");
    }
}

struct ValueRef
{
    std::uint32_t x{}, y{}, z{};
    bool vector{};
};

class GraphCompiler
{
public:
    GraphCompiler(Emitter& emitter,
        const std::unordered_map<std::string, std::uint32_t>& resolvedTextures)
        : emitter_(emitter), resolvedTextures_(resolvedTextures) {}

    void setRemainingUses(std::unordered_map<const mx::Node*, std::uint32_t> uses)
    {
        remainingUses_ = std::move(uses);
    }

    ValueRef input(const mx::NodePtr& node, const char* name, const bool vector,
        const float fallback = 0.0f, const mx::Color3 colorFallback = mx::Color3(0.0f))
    {
        const mx::InputPtr port = node->getInput(name);
        // A material input may target a named output on a MaterialX
        // nodegraph rather than a node in the document graph.  Resolve the
        // output's own upstream node and preserve its output name; treating
        // the nodegraph output name as a node output is invalid (for example
        // NG_spike/base_color_out -> tinted_noise).
        if (port) {
            if (const mx::OutputPtr output = port->getConnectedOutput()) {
                if (const mx::NodePtr connected = output->getConnectedNode()) {
                    const ValueRef value = compileOutput(connected, output->getOutputString());
                    if (!consumptionScopes_.empty())
                        consumptionScopes_.back().push_back(connected.get());
                    return coerce(value, vector);
                }
            }
        }
        if (port && port->getConnectedNode()) {
            const mx::NodePtr connected = port->getConnectedNode();
            const ValueRef value = compileOutput(connected, port->getOutputString());
            // A value remains live until this consuming instruction has been
            // emitted. This is the same last-user lifetime model as Cycles'
            // stack_clear_users(), not a monotonic temporary allocator.
            if (!consumptionScopes_.empty())
                consumptionScopes_.back().push_back(connected.get());
            return coerce(value, vector);
        }
        // Arithmetic nodedefs allow a scalar operand to feed a vector
        // operation (for example the Wave exporter emits vector3 * float).
        // Decode by the port's own declared type, then splat only for the
        // consuming operation; never attempt to read a float Value as color3.
        if (vector && port && (port->getType() == "float" || port->getType() == "integer"
                || port->getType() == "boolean")) {
            const std::uint32_t word = floatWord(floatInput(node, name, fallback));
            return coerce({word, word, word, false}, true);
        }
        if (!vector) {
            float scalar = fallback;
            if (port && (port->getType() == "color3" || port->getType() == "vector3")) {
                const mx::ValuePtr value = node->getInputValue(name);
                if (value) {
                    scalar = port->getType() == "color3"
                        ? value->asA<mx::Color3>()[0] : value->asA<mx::Vector3>()[0];
                }
            } else if (port && port->getType() == "vector2") {
                // e.g. dielectric_bsdf/conductor_bsdf's anisotropic `roughness`
                // fed a vector2: NoorRayCompositeBsdf's lobes only model
                // isotropic roughness, so take the first (U) component.
                const mx::ValuePtr value = node->getInputValue(name);
                if (value)
                    scalar = value->asA<mx::Vector2>()[0];
            } else if (port && port->getType() == "color4") {
                const mx::ValuePtr value = node->getInputValue(name);
                if (value)
                    scalar = value->asA<mx::Color4>()[0];
            } else if (port && port->getType() == "vector4") {
                const mx::ValuePtr value = node->getInputValue(name);
                if (value)
                    scalar = value->asA<mx::Vector4>()[0];
            } else {
                scalar = floatInput(node, name, fallback);
            }
            const std::uint32_t word = floatWord(scalar);
            return {word, word, word, false};
        }
        const mx::Color3 color = colorInput(node, name, colorFallback);
        return {floatWord(color[0]), floatWord(color[1]), floatWord(color[2]), true};
    }

    ValueRef compile(const mx::NodePtr& node)
    {
        const auto found = values_.find(node.get());
        if (found != values_.end())
            return found->second;
        if (!active_.insert(node.get()).second)
            throw SvmCompileError("SVM: cycle at MaterialX node '" + node->getName() + "'");

        consumptionScopes_.emplace_back();

        const std::string& category = node->getCategory();
        const bool vector = node->getType() == "color3" || node->getType() == "vector3";
        ValueRef result{};
        if (category == "add" || category == "subtract" || category == "multiply"
            || category == "divide" || category == "min" || category == "max"
            || category == "modulo" || category == "power" || category == "atan2") {
            result = compileBinary(node, vector, category);
        }
        else if (category == "constant") {
            // ND_constant_*: passes its `value` input straight through.
            result = input(node, "value", vector, 0.0f, mx::Color3(0.0f));
        }
        else if (category == "dot") {
            // ND_dot_*: organizational no-op (IM_dot_*_genglsl is a bare
            // "{{in}}" passthrough); useful only for authoring/graph layout.
            result = input(node, "in", vector);
        }
        else if (category == "and" || category == "or") {
            const ValueRef a = input(node, "in1", false);
            const ValueRef b = input(node, "in2", false);
            const StackOffset out = allocate(1);
            NodeMath instruction{static_cast<std::uint32_t>(
                category == "and" ? MathOp::And : MathOp::Or), a.x, b.x, 0, out};
            emitter_.add(NodeType::Math, instruction);
            result = stack(out, false);
        }
        else if (category == "not") {
            const ValueRef a = input(node, "in", false);
            const StackOffset out = allocate(1);
            NodeMath instruction{static_cast<std::uint32_t>(MathOp::Not), a.x, 0, 0, out};
            emitter_.add(NodeType::Math, instruction);
            result = stack(out, false);
        }
        else if (category == "absval" || category == "floor" || category == "ceil"
            || category == "round" || category == "sign" || category == "sin"
            || category == "cos" || category == "tan" || category == "asin"
            || category == "acos" || category == "ln" || category == "exp"
            || category == "sqrt" || category == "fract") {
            result = compileUnary(node, vector, category);
        }
        else if (category == "mix" || category == "plus" || category == "minus"
            || category == "difference" || category == "burn" || category == "dodge"
            || category == "screen" || category == "overlay") {
            const ValueRef a = input(node, "bg", vector);
            const ValueRef b = input(node, "fg", vector);
            const ValueRef f = input(node, "mix", false);
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeMix instruction{};
            instruction.fac = f.x;
            instruction.value1X = a.x; instruction.value1Y = a.y; instruction.value1Z = a.z;
            instruction.value2X = b.x; instruction.value2Y = b.y; instruction.value2Z = b.z;
            instruction.resultOffset = out;
            instruction.isColor = vector ? 1 : 0;
            instruction.blendType = category == "plus" ? MixBlendType::Add
                : category == "minus" ? MixBlendType::Subtract
                : category == "difference" ? MixBlendType::Difference
                : category == "burn" ? MixBlendType::Burn
                : category == "dodge" ? MixBlendType::Dodge
                : category == "screen" ? MixBlendType::Screen
                : category == "overlay" ? MixBlendType::Overlay : MixBlendType::Blend;
            emitter_.add(NodeType::Mix, instruction);
            result = stack(out, vector);
        }
        else if (category == "ifgreater" || category == "ifgreatereq" || category == "ifequal") {
            // MaterialX conditional nodes lower to the same math+mix pair
            // Cycles uses for value-driven selections. This stays branchless
            // in the interpreter and only needs one transient scalar slot.
            const StackOffset condition = allocate(1);
            NodeMath compare{};
            compare.mathType = static_cast<std::uint32_t>(category == "ifgreater"
                ? MathOp::GreaterThan : category == "ifgreatereq" ? MathOp::GreaterEqual : MathOp::Equal);
            compare.value1 = input(node, "value1", false).x;
            compare.value2 = input(node, "value2", false).x;
            compare.resultOffset = condition;
            emitter_.add(NodeType::Math, compare);

            const ValueRef falseValue = input(node, "in2", vector);
            const ValueRef trueValue = input(node, "in1", vector);
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeMix select{};
            select.fac = encodeStackOffset(condition);
            select.value1X = falseValue.x; select.value1Y = falseValue.y; select.value1Z = falseValue.z;
            select.value2X = trueValue.x; select.value2Y = trueValue.y; select.value2Z = trueValue.z;
            select.resultOffset = out;
            select.isColor = vector ? 1 : 0;
            emitter_.add(NodeType::Mix, select);
            result = stack(out, vector);
        }
        else if (category == "magnitude") {
            // ND_magnitude_vector{2,3,4}: the exporter supplies vector3.
            // Keep it a VectorMath instruction, matching Cycles' vector
            // math lowering, rather than folding it on the host.
            const ValueRef value = input(node, "in", true);
            const StackOffset out = allocate(1);
            NodeVectorMath instruction{};
            instruction.mathType = static_cast<std::uint32_t>(VectorMathOp::Magnitude);
            instruction.value1X = value.x;
            instruction.value1Y = value.y;
            instruction.value1Z = value.z;
            instruction.resultOffset = out;
            instruction.resultIsScalar = 1;
            emitter_.add(NodeType::VectorMath, instruction);
            result = stack(out, false);
        }
        else if (category == "dotproduct" || category == "crossproduct" || category == "distance"
            || category == "normalize" || category == "reflect" || category == "refract") {
            const ValueRef a = input(node,
                (category == "dotproduct" || category == "crossproduct" || category == "distance")
                    ? "in1" : "in", true);
            const ValueRef b = (category == "dotproduct" || category == "crossproduct" || category == "distance") ? input(node, "in2", true)
                : (category == "reflect" || category == "refract") ? input(node, "normal", true)
                                                                        : ValueRef{};
            const bool scalarResult = category == "dotproduct" || category == "distance";
            const StackOffset out = allocate(scalarResult ? 1 : 3);
            NodeVectorMath instruction{};
            instruction.mathType = static_cast<std::uint32_t>(category == "dotproduct" ? VectorMathOp::DotProduct
                : category == "distance" ? VectorMathOp::Distance
                : category == "crossproduct" ? VectorMathOp::CrossProduct
                : category == "reflect" ? VectorMathOp::Reflect
                : category == "refract" ? VectorMathOp::Refract : VectorMathOp::Normalize);
            instruction.value1X = a.x; instruction.value1Y = a.y; instruction.value1Z = a.z;
            instruction.value2X = b.x; instruction.value2Y = b.y; instruction.value2Z = b.z;
            if (category == "refract")
                instruction.value3X = input(node, "ior", false, 1.0f).x;
            instruction.resultOffset = out;
            instruction.resultIsScalar = scalarResult ? 1 : 0;
            emitter_.add(NodeType::VectorMath, instruction);
            result = stack(out, !scalarResult);
        }
        else if (category == "combine2" || category == "combine3") {
            const StackOffset out = allocate(3);
            NodeCombineColor instruction{};
            instruction.x = input(node, "in1", false).x;
            instruction.y = input(node, "in2", false).x;
            instruction.z = category == "combine3" ? input(node, "in3", false).x : floatWord(0.0f);
            instruction.layout = ColorChannelLayout::Rgb;
            instruction.resultOffset = out;
            emitter_.add(NodeType::CombineColor, instruction);
            result = stack(out, true);
        }
        else if (category == "separate2" || category == "separate3") {
            // The standard library lowers these to one extract per channel.
            // Retain Cycles' one multi-result separate instruction and expose
            // MaterialX's named output ports from the same stack allocation.
            const ValueRef value = input(node, "in", true);
            const StackOffset out = allocate(3);
            NodeSeparateColor instruction{};
            instruction.colorX = value.x; instruction.colorY = value.y; instruction.colorZ = value.z;
            instruction.layout = ColorChannelLayout::Rgb;
            instruction.resultXOffset = out;
            instruction.resultYOffset = static_cast<StackOffset>(out + 1);
            instruction.resultZOffset = category == "separate3"
                ? static_cast<StackOffset>(out + 2) : InvalidOffset;
            emitter_.add(NodeType::SeparateColor, instruction);
            result = stack(out, true);
            auto& outputs = namedOutputs_[node.get()];
            const mx::InputPtr inputPort = node->getInput("in");
            if (inputPort && inputPort->getType() == "color3") {
                outputs.emplace("outr", stack(out, false));
                outputs.emplace("outg", stack(static_cast<StackOffset>(out + 1), false));
                if (category == "separate3")
                    outputs.emplace("outb", stack(static_cast<StackOffset>(out + 2), false));
            }
            else {
                outputs.emplace("outx", stack(out, false));
                outputs.emplace("outy", stack(static_cast<StackOffset>(out + 1), false));
                if (category == "separate3")
                    outputs.emplace("outz", stack(static_cast<StackOffset>(out + 2), false));
            }
        }
        else if (category == "remap" || category == "smoothstep") {
            const StackOffset out = allocate(1);
            NodeRemapRange instruction{};
            instruction.value = input(node, "in", false).x;
            instruction.inLow = input(node, "inlow", false, 0.0f).x;
            instruction.inHigh = input(node, "inhigh", false, 1.0f).x;
            instruction.outLow = input(node, "outlow", false, 0.0f).x;
            instruction.outHigh = input(node, "outhigh", false, 1.0f).x;
            instruction.resultOffset = out;
            instruction.smoothstep = category == "smoothstep" ? 1 : 0;
            if (category == "smoothstep") {
                instruction.inLow = input(node, "low", false, 0.0f).x;
                instruction.inHigh = input(node, "high", false, 1.0f).x;
            }
            emitter_.add(NodeType::RemapRange, instruction);
            result = stack(out, false);
        }
        else if (category == "range") {
            // Direct port of NG_range_*: remap to [0,1], apply signed
            // gamma, remap to the output range, then select optional clamp.
            // A single opcode avoids the standard graph's six intermediate
            // value nodes and their associated stack lifetimes.
            const ValueRef value = input(node, "in", vector);
            const ValueRef inLow = input(node, "inlow", vector, 0.0f, mx::Color3(0.0f));
            const ValueRef inHigh = input(node, "inhigh", vector, 1.0f, mx::Color3(1.0f));
            const ValueRef gamma = input(node, "gamma", vector, 1.0f, mx::Color3(1.0f));
            const ValueRef outLow = input(node, "outlow", vector, 0.0f, mx::Color3(0.0f));
            const ValueRef outHigh = input(node, "outhigh", vector, 1.0f, mx::Color3(1.0f));
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeRange instruction{};
            instruction.valueX = value.x; instruction.valueY = value.y; instruction.valueZ = value.z;
            instruction.inLowX = inLow.x; instruction.inLowY = inLow.y; instruction.inLowZ = inLow.z;
            instruction.inHighX = inHigh.x; instruction.inHighY = inHigh.y; instruction.inHighZ = inHigh.z;
            instruction.gammaX = gamma.x; instruction.gammaY = gamma.y; instruction.gammaZ = gamma.z;
            instruction.outLowX = outLow.x; instruction.outLowY = outLow.y; instruction.outLowZ = outLow.z;
            instruction.outHighX = outHigh.x; instruction.outHighY = outHigh.y; instruction.outHighZ = outHigh.z;
            instruction.doClamp = input(node, "doclamp", false, 0.0f).x;
            instruction.resultOffset = out;
            instruction.isColor = vector ? 1 : 0;
            emitter_.add(NodeType::Range, instruction);
            result = stack(out, vector);
        }
        else if (category == "hsvadjust") {
            // MaterialX's functional hsvadjust graph matches Cycles' compact
            // NODE_HSV operation. Keep it as one opcode rather than expand
            // rgbtohsv/multiply/add/hsvtorgb into temporary stack values.
            const ValueRef color = input(node, "in", true);
            const ValueRef amount = input(node, "amount", true);
            const StackOffset out = allocate(3);
            NodeHsvAdjust instruction{};
            instruction.hue = amount.x;
            instruction.saturation = amount.y;
            instruction.value = amount.z;
            instruction.fac = floatWord(1.0f);
            instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
            instruction.resultOffset = out;
            emitter_.add(NodeType::HsvAdjust, instruction);
            result = stack(out, true);
        }
        else if (category == "invert") {
            // MaterialX invert is `amount - in`, not Blender's fac-mixed
            // invert. Keep its vector/scalar variants in the existing opcode.
            const ValueRef value = input(node, "in", vector);
            const ValueRef amount = input(node, "amount", vector, 1.0f,
                mx::Color3(1.0f));
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeInvert instruction{};
            instruction.colorX = value.x; instruction.colorY = value.y; instruction.colorZ = value.z;
            instruction.amountX = amount.x; instruction.amountY = amount.y; instruction.amountZ = amount.z;
            instruction.resultOffset = out;
            instruction.isColor = vector ? 1 : 0;
            emitter_.add(NodeType::Invert, instruction);
            result = stack(out, vector);
        }
        else if (category == "contrast") {
            const ValueRef value = input(node, "in", vector);
            const ValueRef amount = input(node, "amount", vector, 1.0f,
                mx::Color3(1.0f));
            const ValueRef pivot = input(node, "pivot", vector, 0.5f,
                mx::Color3(0.5f));
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeContrast instruction{};
            instruction.valueX = value.x; instruction.valueY = value.y; instruction.valueZ = value.z;
            instruction.amountX = amount.x; instruction.amountY = amount.y; instruction.amountZ = amount.z;
            instruction.pivotX = pivot.x; instruction.pivotY = pivot.y; instruction.pivotZ = pivot.z;
            instruction.resultOffset = out;
            instruction.isColor = vector ? 1 : 0;
            emitter_.add(NodeType::Contrast, instruction);
            result = stack(out, vector);
        }
        else if (category == "saturate") {
            const ValueRef color = input(node, "in", true);
            const StackOffset out = allocate(3);
            NodeSaturate instruction{};
            instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
            instruction.amount = input(node, "amount", false, 1.0f).x;
            const ValueRef luma = input(node, "lumacoeffs", true, 0.0f,
                mx::Color3(0.2722287f, 0.6740818f, 0.0536895f));
            instruction.lumaX = luma.x; instruction.lumaY = luma.y; instruction.lumaZ = luma.z;
            instruction.resultOffset = out;
            emitter_.add(NodeType::Saturate, instruction);
            result = stack(out, true);
        }
        else if (category == "rgbtohsv") {
            const ValueRef color = input(node, "in", true);
            const StackOffset out = allocate(3);
            NodeSeparateColor instruction{};
            instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
            instruction.layout = ColorChannelLayout::Hsv;
            instruction.resultXOffset = out;
            instruction.resultYOffset = static_cast<StackOffset>(out + 1);
            instruction.resultZOffset = static_cast<StackOffset>(out + 2);
            emitter_.add(NodeType::SeparateColor, instruction);
            result = stack(out, true);
        }
        else if (category == "hsvtorgb") {
            const ValueRef hsv = input(node, "in", true);
            const StackOffset out = allocate(3);
            NodeCombineColor instruction{};
            instruction.x = hsv.x; instruction.y = hsv.y; instruction.z = hsv.z;
            instruction.layout = ColorChannelLayout::Hsv;
            instruction.resultOffset = out;
            emitter_.add(NodeType::CombineColor, instruction);
            result = stack(out, true);
        }
        else if (category == "clamp") {
            const ValueRef value = input(node, "in", vector);
            const ValueRef low = input(node, "low", vector, 0.0f, mx::Color3(0.0f));
            const ValueRef high = input(node, "high", vector, 1.0f, mx::Color3(1.0f));
            const StackOffset out = allocate(vector ? 3 : 1);
            const std::uint32_t values[3]{value.x, value.y, value.z};
            const std::uint32_t lows[3]{low.x, low.y, low.z};
            const std::uint32_t highs[3]{high.x, high.y, high.z};
            for (int i = 0; i < (vector ? 3 : 1); ++i) {
                NodeClamp instruction{};
                instruction.value = values[i];
                instruction.minValue = lows[i];
                instruction.maxValue = highs[i];
                instruction.resultOffset = static_cast<StackOffset>(out + i);
                emitter_.add(NodeType::Clamp, instruction);
            }
            result = stack(out, vector);
        }
        else if (category == "texcoord" || category == "position" || category == "normal"
            || category == "tangent" || category == "bitangent" || category == "geomcolor"
            || category == "viewdirection") {
            const StackOffset out = allocate(3);
            NodeTexCoord instruction{};
            instruction.source = category == "texcoord" ? TexCoordSource::UV
                : category == "normal" ? TexCoordSource::Normal
                : category == "tangent" ? TexCoordSource::Tangent
                : category == "bitangent" ? TexCoordSource::Bitangent
                : category == "geomcolor" ? TexCoordSource::VertexColor
                : category == "viewdirection" ? TexCoordSource::ViewDirection
                : TexCoordSource::Object;
            instruction.resultOffset = out;
            emitter_.add(NodeType::TexCoord, instruction);
            result = stack(out, true);
        }
        else if (category == "rotate3d") {
            // MaterialX rotate3d is an axis-angle operation. Keep it as a
            // single semantic instruction; expanding the reference graph
            // would cost several VectorMath temporaries per rotation.
            const ValueRef in = input(node, "in", true);
            const ValueRef axis = input(node, "axis", true, 0.0f, mx::Color3(0.0f, 1.0f, 0.0f));
            const StackOffset out = allocate(3);
            NodeRotate3d instruction{};
            instruction.inX = in.x; instruction.inY = in.y; instruction.inZ = in.z;
            instruction.amountDegrees = input(node, "amount", false).x;
            instruction.axisX = axis.x; instruction.axisY = axis.y; instruction.axisZ = axis.z;
            instruction.resultOffset = out;
            emitter_.add(NodeType::Rotate3d, instruction);
            result = stack(out, true);
        }
        else if (category == "image") {
            const mx::InputPtr file = node->getInput("file");
            if (!file || file->getValueString().empty())
                throw SvmCompileError("SVM: image node '" + node->getName() + "' has no file");
            const auto texture = resolvedTextures_.find(file->getValueString());
            if (texture == resolvedTextures_.end())
                throw SvmCompileError("SVM: unresolved image '" + file->getValueString() + "'");
            const StackOffset out = allocate(3);
            const StackOffset alpha = allocate(1);
            NodeImageTexture instruction{};
            const auto [slot, inserted] = textureSlots_.emplace(texture->second,
                static_cast<std::uint32_t>(textureSlots_.size()));
            if (inserted)
                textures_.push_back(texture->second);
            const ValueRef uv = input(node, "texcoord", true, 0.0f, mx::Color3(0.0f));
            instruction.textureSlot = static_cast<std::int32_t>(slot->second);
            instruction.uvX = uv.x; instruction.uvY = uv.y;
            instruction.resultColorOffset = out;
            instruction.resultAlphaOffset = alpha;
            const auto addressMode = [&](const char* name) -> StackOffset {
                const mx::InputPtr port = node->getInput(name);
                const std::string value = port ? port->getValueString() : "periodic";
                return value == "clamp" ? 1 : value == "mirror" ? 2 : value == "constant" ? 3 : 0;
            };
            instruction.uAddressMode = addressMode("uaddressmode");
            instruction.vAddressMode = addressMode("vaddressmode");
            const mx::InputPtr filter = node->getInput("filtertype");
            instruction.filterType = filter && filter->getValueString() == "closest" ? 1 : 0;
            emitter_.add(NodeType::ImageTexture, instruction);
            result = stack(out, true);
            namedOutputs_[node.get()].emplace("out", result);
            namedOutputs_[node.get()].emplace("alpha", stack(alpha, false));
        }
        else if (category == "normalmap") {
            const StackOffset out = allocate(3);
            const ValueRef color = input(node, "in", true, 0.0f, mx::Color3(0.5f, 0.5f, 1.0f));
            const mx::InputPtr scalePort = node->getInput("scale");
            const bool vectorScale = scalePort && scalePort->getType() == "vector2";
            const ValueRef scale = input(node, "scale", vectorScale, 1.0f, mx::Color3(1.0f));
            const auto geometry = [&](const char* name, const TexCoordSource fallback,
                                      std::uint32_t& x, std::uint32_t& y, std::uint32_t& z) {
                const mx::InputPtr port = node->getInput(name);
                if (port && port->getConnectedNode()) {
                    const ValueRef value = input(node, name, true);
                    x = value.x; y = value.y; z = value.z;
                }
                else {
                    const StackOffset generated = allocate(3);
                    emitter_.add(NodeType::TexCoord, NodeTexCoord{fallback, generated});
                    const ValueRef value = stack(generated, true);
                    x = value.x; y = value.y; z = value.z;
                }
            };
            NodeNormalMap instruction{};
            instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
            instruction.scaleX = scale.x;
            instruction.scaleY = vectorScale ? scale.y : scale.x;
            geometry("normal", TexCoordSource::Normal,
                instruction.normalX, instruction.normalY, instruction.normalZ);
            geometry("tangent", TexCoordSource::Tangent,
                instruction.tangentX, instruction.tangentY, instruction.tangentZ);
            geometry("bitangent", TexCoordSource::Bitangent,
                instruction.bitangentX, instruction.bitangentY, instruction.bitangentZ);
            instruction.resultOffset = out;
            emitter_.add(NodeType::NormalMap, instruction);
            result = stack(out, true);
        }
        else if (category == "convert") {
            // MaterialX's convert node changes the port type, not the
            // underlying computation.  The Blender exporter deliberately
            // lowers color/vector -> float conversions to luminance or
            // magnitude first, so the remaining generic conversions are
            // scalar splats and color/vector shape conversions.  SVM keeps
            // those as the same three component values; coerce performs the
            // scalar splat when the destination is a vector/color.
            result = input(node, "in", vector);
        }
        else if (category == "luminance") {
            // Same compact dot-product representation Cycles uses for color
            // to value operations; MaterialX defines Rec.709 luminance.
            const ValueRef color = input(node, "in", true);
            const StackOffset out = allocate(1);
            NodeVectorMath instruction{};
            instruction.mathType = static_cast<std::uint32_t>(VectorMathOp::DotProduct);
            instruction.value1X = color.x; instruction.value1Y = color.y; instruction.value1Z = color.z;
            instruction.value2X = floatWord(0.2126f);
            instruction.value2Y = floatWord(0.7152f);
            instruction.value2Z = floatWord(0.0722f);
            instruction.resultOffset = out;
            instruction.resultIsScalar = 1;
            emitter_.add(NodeType::VectorMath, instruction);
            result = stack(out, false);
        }
        else if (category == "blackbody") {
            const StackOffset out = allocate(3);
            NodeBlackbody instruction{};
            instruction.temperature = input(node, "temperature", false, 6500.0f).x;
            instruction.resultOffset = out;
            emitter_.add(NodeType::Blackbody, instruction);
            result = stack(out, true);
        }
        else if (category == "extract") {
            // MaterialX extract is the separate-color operation with exactly
            // one requested output.  Blender emits a literal index for every
            // multi-output connection it lowers.
            const int index = static_cast<int>(floatInput(node, "index", 0.0f));
            if (index == 3) {
                const mx::InputPtr inputPort = node->getInput("in");
                const mx::NodePtr source = inputPort ? inputPort->getConnectedNode() : nullptr;
                if (!source || source->getCategory() != "image")
                    throw SvmCompileError("SVM: extract component 3 requires a MaterialX image node ('"
                        + node->getName() + "')");
                // Blender's image Alpha socket exports as extract(image, 3).
                // The image opcode writes alpha alongside RGB, so retain one
                // texture fetch instead of inventing a fourth generic value
                // component for every SVM instruction.
                result = compileOutput(source, "alpha");
            }
            else {
                if (index < 0 || index > 2)
                    throw SvmCompileError("SVM: extract index out of range on MaterialX node '" + node->getName() + "'");
            const ValueRef in = input(node, "in", true);
            const StackOffset out = allocate(1);
            NodeSeparateColor instruction{};
            instruction.colorX = in.x; instruction.colorY = in.y; instruction.colorZ = in.z;
            instruction.layout = ColorChannelLayout::Rgb;
            instruction.resultXOffset = index == 0 ? out : InvalidOffset;
            instruction.resultYOffset = index == 1 ? out : InvalidOffset;
            instruction.resultZOffset = index == 2 ? out : InvalidOffset;
            emitter_.add(NodeType::SeparateColor, instruction);
            result = stack(out, false);
            }
        }
        else if (category == "fractal2d" || category == "fractal3d") {
            // MaterialX stdlib: sum Perlin octaves at progressively higher
            // frequency and lower amplitude.  This is a MaterialX node, not
            // Cycles' Noise Texture, so retain all four standard inputs.
            const bool is2d = category == "fractal2d";
            const ValueRef position = is2d
                ? input(node, "texcoord", true, 0.0f, mx::Color3(0.0f))
                : input(node, "position", true, 0.0f, mx::Color3(0.0f));
            const ValueRef amplitude = input(node, "amplitude", vector, 1.0f, mx::Color3(1.0f));
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeProceduralTexture instruction{};
            instruction.posX = position.x;
            instruction.posY = position.y;
            instruction.posZ = position.z;
            instruction.amplitudeX = amplitude.x;
            instruction.amplitudeY = amplitude.y;
            instruction.amplitudeZ = amplitude.z;
            instruction.octaves = input(node, "octaves", false, 3.0f).x;
            instruction.lacunarity = input(node, "lacunarity", false, 2.0f).x;
            instruction.diminish = input(node, "diminish", false, 0.5f).x;
            instruction.resultOffset = out;
            instruction.resultIsVector = vector ? 1 : 0;
            instruction.is2d = is2d ? 1 : 0;
            emitter_.add(NodeType::FractalNoiseTexture, instruction);
            result = stack(out, vector);
        }
        else if (category == "noise2d" || category == "noise3d") {
            // mx_noise3d_* is a single MaterialX Perlin evaluation followed
            // by amplitude/pivot. Reuse the same compact payload style as
            // fractal3d while keeping its non-octave semantics distinct.
            const bool is2d = category == "noise2d";
            const ValueRef position = is2d
                ? input(node, "texcoord", true, 0.0f, mx::Color3(0.0f))
                : input(node, "position", true, 0.0f, mx::Color3(0.0f));
            const ValueRef amplitude = input(node, "amplitude", vector, 1.0f, mx::Color3(1.0f));
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeNoiseTexture instruction{};
            instruction.posX = position.x; instruction.posY = position.y; instruction.posZ = position.z;
            instruction.amplitudeX = amplitude.x;
            instruction.amplitudeY = amplitude.y;
            instruction.amplitudeZ = amplitude.z;
            instruction.pivot = input(node, "pivot", false, 0.0f).x;
            instruction.resultOffset = out;
            instruction.resultIsVector = vector ? 1 : 0;
            instruction.is2d = is2d ? 1 : 0;
            emitter_.add(NodeType::NoiseTexture, instruction);
            result = stack(out, vector);
        }
        else if (category == "worleynoise2d" || category == "worleynoise3d") {
            const bool is2d = category == "worleynoise2d";
            const ValueRef position = is2d
                ? input(node, "texcoord", true, 0.0f, mx::Color3(0.0f))
                : input(node, "position", true, 0.0f, mx::Color3(0.0f));
            const StackOffset out = allocate(vector ? 3 : 1);
            NodeWorleyNoiseTexture instruction{};
            instruction.posX = position.x; instruction.posY = position.y; instruction.posZ = position.z;
            instruction.jitter = input(node, "jitter", false, 1.0f).x;
            instruction.style = input(node, "style", false, 0.0f).x;
            instruction.resultOffset = out;
            instruction.resultIsVector = vector ? 1 : 0;
            instruction.is2d = is2d ? 1 : 0;
            emitter_.add(NodeType::WorleyNoiseTexture, instruction);
            result = stack(out, vector);
        }
        else if (category == "cellnoise2d" || category == "cellnoise3d") {
            const ValueRef position = category == "cellnoise2d"
                ? input(node, "texcoord", true, 0.0f, mx::Color3(0.0f))
                : input(node, "position", true, 0.0f, mx::Color3(0.0f));
            const StackOffset out = allocate(1);
            NodeCellNoiseTexture instruction{};
            instruction.posX = position.x; instruction.posY = position.y; instruction.posZ = position.z;
            instruction.resultOffset = out;
            instruction.is2d = category == "cellnoise2d" ? 1 : 0;
            emitter_.add(NodeType::CellNoiseTexture, instruction);
            result = stack(out, false);
        }
        else if (category == "unifiednoise2d" || category == "unifiednoise3d") {
            // Exact semantic collapse of NG_unifiednoise*_float from
            // stdlib_ng.mtlx.  Do not emit its implementation graph: that
            // graph is deliberately a portable reference, while the SVM is
            // the fixed interpreter implementation of the same operation.
            const bool is2d = category == "unifiednoise2d";
            const ValueRef position = is2d
                ? input(node, "texcoord", true, 0.0f, mx::Color3(0.0f))
                : input(node, "position", true, 0.0f, mx::Color3(0.0f));
            const ValueRef frequency = input(node, "freq", true, 1.0f, mx::Color3(1.0f));
            const ValueRef coordinateOffset = input(node, "offset", true, 0.0f, mx::Color3(0.0f));
            const StackOffset out = allocate(1);
            NodeUnifiedNoiseTexture instruction{};
            instruction.posX = position.x; instruction.posY = position.y; instruction.posZ = position.z;
            instruction.freqX = frequency.x; instruction.freqY = frequency.y; instruction.freqZ = frequency.z;
            instruction.offsetX = coordinateOffset.x; instruction.offsetY = coordinateOffset.y; instruction.offsetZ = coordinateOffset.z;
            instruction.jitter = input(node, "jitter", false, 1.0f).x;
            instruction.outMin = input(node, "outmin", false, 0.0f).x;
            instruction.outMax = input(node, "outmax", false, 1.0f).x;
            instruction.clampOutput = input(node, "clampoutput", false, 1.0f).x;
            instruction.octaves = input(node, "octaves", false, 3.0f).x;
            instruction.lacunarity = input(node, "lacunarity", false, 2.0f).x;
            instruction.diminish = input(node, "diminish", false, 0.5f).x;
            instruction.noiseType = input(node, "type", false, 0.0f).x;
            instruction.style = input(node, "style", false, 0.0f).x;
            instruction.resultOffset = out;
            instruction.is2d = is2d ? 1 : 0;
            emitter_.add(NodeType::UnifiedNoiseTexture, instruction);
            result = stack(out, false);
        }
        else if (category == "checkerboard") {
            // Direct collapse of NG_checkerboard_color3's multiply,
            // subtract, floor, dot, modulo and mix sequence.
            const ValueRef color1 = input(node, "color1", true, 0.0f, mx::Color3(1.0f));
            const ValueRef color2 = input(node, "color2", true, 0.0f, mx::Color3(0.0f));
            const ValueRef tiling = input(node, "uvtiling", true, 0.0f,
                mx::Color3(8.0f, 8.0f, 0.0f));
            const ValueRef offset = input(node, "uvoffset", true, 0.0f, mx::Color3(0.0f));
            const ValueRef uv = input(node, "texcoord", true, 0.0f, mx::Color3(0.0f));
            const StackOffset out = allocate(3);
            NodeCheckerTexture instruction{};
            instruction.color1X = color1.x; instruction.color1Y = color1.y; instruction.color1Z = color1.z;
            instruction.color2X = color2.x; instruction.color2Y = color2.y; instruction.color2Z = color2.z;
            instruction.tilingX = tiling.x; instruction.tilingY = tiling.y;
            instruction.offsetX = offset.x; instruction.offsetY = offset.y;
            instruction.uvX = uv.x; instruction.uvY = uv.y;
            instruction.resultOffset = out;
            emitter_.add(NodeType::CheckerTexture, instruction);
            result = stack(out, true);
        }
        else if (category == "roughness_anisotropy") {
            // Direct, branchless port of mx_roughness_anisotropy.glsl: the
            // conditional there (anisotropy > 0 ? ... : isotropic) collapses
            // to the same expression once anisotropy is clamped to [0, 0.98]
            // first, since aspect == 1 whenever anisotropy <= 0.
            const ValueRef roughness = input(node, "roughness", false, 0.0f);
            const ValueRef anisotropy = input(node, "anisotropy", false, 0.0f);
            const StackOffset roughnessSqr = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::Multiply),
                roughness.x, roughness.x, 0, roughnessSqr});
            emitter_.add(NodeType::Clamp, NodeClamp{encodeStackOffset(roughnessSqr),
                floatWord(1e-4f), floatWord(1.0f), roughnessSqr});
            const StackOffset anisotropyClamped = allocate(1);
            emitter_.add(NodeType::Clamp, NodeClamp{anisotropy.x, floatWord(0.0f), floatWord(0.98f),
                anisotropyClamped});
            const StackOffset aspect = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::Subtract),
                floatWord(1.0f), encodeStackOffset(anisotropyClamped), 0, aspect});
            emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::Sqrt),
                encodeStackOffset(aspect), 0, 0, aspect});
            const StackOffset out = allocate(3);
            const StackOffset roughnessOverAspect = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::Divide),
                encodeStackOffset(roughnessSqr), encodeStackOffset(aspect), 0, roughnessOverAspect});
            emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::Min),
                encodeStackOffset(roughnessOverAspect), floatWord(1.0f), 0, out});
            emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::Multiply),
                encodeStackOffset(roughnessSqr), encodeStackOffset(aspect), 0,
                static_cast<StackOffset>(out + 1)});
            result = stack(out, true);
        }
        else if (category == "roughness_dual") {
            // Direct port of mx_roughness_dual.glsl.  roughness.y < 0 means
            // "use roughness.x for both axes"; select branchlessly with the
            // same Math(compare) + Mix pattern ifgreater/ifequal use above.
            const ValueRef roughness = input(node, "roughness", true, 0.0f, mx::Color3(0.0f));
            const StackOffset useX = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::GreaterThan),
                floatWord(0.0f), roughness.y, 0, useX});
            const StackOffset y = allocate(1);
            emitter_.add(NodeType::Mix, NodeMix{encodeStackOffset(useX),
                roughness.y, 0, 0, roughness.x, 0, 0, y, 0});
            const StackOffset out = allocate(3);
            const std::uint32_t values[2]{roughness.x, encodeStackOffset(y)};
            for (int i = 0; i < 2; ++i) {
                const StackOffset squared = allocate(1);
                emitter_.add(NodeType::Math, NodeMath{static_cast<std::uint32_t>(MathOp::Multiply),
                    values[i], values[i], 0, squared});
                emitter_.add(NodeType::Clamp, NodeClamp{encodeStackOffset(squared),
                    floatWord(1e-4f), floatWord(1.0f), static_cast<StackOffset>(out + i)});
            }
            result = stack(out, true);
        }
        else if (category == "artistic_ior") {
            // ND_artistic_ior has two named outputs.  Keep the MaterialX
            // standard-library operation intact as one SVM instruction,
            // exactly as Cycles keeps multi-result texture nodes compact.
            const ValueRef reflectivity = input(node, "reflectivity", true, 0.0f, mx::Color3(0.5f));
            const ValueRef edgeColor = input(node, "edge_color", true, 0.0f, mx::Color3(1.0f));
            ValueRef extinction{};
            result = artisticIor(reflectivity, edgeColor, extinction);
            namedOutputs_[node.get()].emplace("ior", result);
            namedOutputs_[node.get()].emplace("extinction", extinction);
        }
        else {
            throw SvmCompileError("SVM: unsupported MaterialX value node '" + category
                + "' ('" + node->getName() + "')");
        }
        active_.erase(node.get());
        values_.emplace(node.get(), result);
        const std::vector<const mx::Node*> consumed = std::move(consumptionScopes_.back());
        consumptionScopes_.pop_back();
        for (const mx::Node* consumedNode : consumed)
            consume(consumedNode);
        return result;
    }

    const std::vector<std::uint32_t>& textures() const { return textures_; }

    ValueRef multiply(const ValueRef a, const ValueRef b)
    {
        const bool vector = a.vector || b.vector;
        const ValueRef av = coerce(a, vector);
        const ValueRef bv = coerce(b, vector);
        const StackOffset out = allocate(vector ? 3 : 1);
        const std::uint32_t aa[3]{av.x, av.y, av.z};
        const std::uint32_t bb[3]{bv.x, bv.y, bv.z};
        for (int i = 0; i < (vector ? 3 : 1); ++i) {
            NodeMath instruction{static_cast<std::uint32_t>(MathOp::Multiply), aa[i], bb[i], 0,
                static_cast<StackOffset>(out + i)};
            emitter_.add(NodeType::Math, instruction);
        }
        return stack(out, vector);
    }

    ValueRef oneMinus(const ValueRef input)
    {
        const StackOffset out = allocate(1);
        NodeMath instruction{static_cast<std::uint32_t>(MathOp::Subtract),
            floatWord(1.0f), input.x, 0, out};
        emitter_.add(NodeType::Math, instruction);
        return stack(out, false);
    }

    ValueRef clamp01(const ValueRef input)
    {
        const StackOffset out = allocate(1);
        emitter_.add(NodeType::Clamp,
            NodeClamp{input.x, floatWord(0.0f), floatWord(1.0f), out});
        return stack(out, false);
    }

    // Shared by the artistic_ior value node and generalized_schlick_bsdf's
    // Fresnel-curve -> conductor IOR/extinction mapping (both solve the same
    // ND_artistic_ior equation for a reflectivity/edge-tint pair).
    ValueRef artisticIor(const ValueRef& reflectivity, const ValueRef& edgeTint, ValueRef& outExtinction)
    {
        const StackOffset ior = allocate(3);
        const StackOffset extinction = allocate(3);
        NodeArtisticIor instruction{};
        instruction.reflectivityX = reflectivity.x; instruction.reflectivityY = reflectivity.y;
        instruction.reflectivityZ = reflectivity.z;
        instruction.edgeTintX = edgeTint.x; instruction.edgeTintY = edgeTint.y; instruction.edgeTintZ = edgeTint.z;
        instruction.resultIorX = encodeStackOffset(ior);
        instruction.resultIorY = encodeStackOffset(static_cast<StackOffset>(ior + 1));
        instruction.resultIorZ = encodeStackOffset(static_cast<StackOffset>(ior + 2));
        instruction.resultExtinctionX = encodeStackOffset(extinction);
        instruction.resultExtinctionY = encodeStackOffset(static_cast<StackOffset>(extinction + 1));
        instruction.resultExtinctionZ = encodeStackOffset(static_cast<StackOffset>(extinction + 2));
        emitter_.add(NodeType::ArtisticIor, instruction);
        outExtinction = stack(extinction, true);
        return stack(ior, true);
    }

private:
    ValueRef compileUnary(const mx::NodePtr& node, const bool vector, const std::string& category)
    {
        const ValueRef inputValue = input(node, "in", vector);
        const MathOp operation = category == "absval" ? MathOp::Absval
            : category == "floor" ? MathOp::Floor : category == "ceil" ? MathOp::Ceil
            : category == "round" ? MathOp::Round : category == "sign" ? MathOp::Sign
            : category == "sin" ? MathOp::Sin : category == "cos" ? MathOp::Cos
            : category == "tan" ? MathOp::Tan : category == "asin" ? MathOp::Asin
            : category == "acos" ? MathOp::Acos : category == "ln" ? MathOp::Ln
            : category == "exp" ? MathOp::Exp : category == "fract" ? MathOp::Fract : MathOp::Sqrt;
        if (!vector) {
            const StackOffset out = allocate(1);
            NodeMath instruction{static_cast<std::uint32_t>(operation), inputValue.x, 0, 0, out};
            emitter_.add(NodeType::Math, instruction);
            return stack(out, false);
        }
        // Cycles emits one scalar operation per component for these MaterialX
        // vector categories; retaining that representation keeps stack use
        // and literal encoding identical to the scalar path.
        const StackOffset out = allocate(3);
        const std::uint32_t inputs[3]{inputValue.x, inputValue.y, inputValue.z};
        for (int i = 0; i < 3; ++i) {
            NodeMath instruction{static_cast<std::uint32_t>(operation), inputs[i], 0, 0,
                static_cast<StackOffset>(out + i)};
            emitter_.add(NodeType::Math, instruction);
        }
        return stack(out, true);
    }

    ValueRef compileBinary(const mx::NodePtr& node, const bool vector, const std::string& category)
    {
        const ValueRef a = input(node, "in1", vector);
        const ValueRef b = input(node, "in2", vector);
        const MathOp operation = category == "add" ? MathOp::Add
            : category == "subtract" ? MathOp::Subtract : category == "multiply" ? MathOp::Multiply
            : category == "divide" ? MathOp::Divide : category == "min" ? MathOp::Min
            : category == "max" ? MathOp::Max : category == "modulo" ? MathOp::Modulo
            : category == "atan2" ? MathOp::Atan2 : MathOp::Power;
        if (!vector) {
            const StackOffset out = allocate(1);
            NodeMath instruction{static_cast<std::uint32_t>(operation), a.x, b.x, 0, out};
            emitter_.add(NodeType::Math, instruction);
            return stack(out, false);
        }
        const StackOffset out = allocate(3);
        const std::uint32_t av[3]{a.x, a.y, a.z};
        const std::uint32_t bv[3]{b.x, b.y, b.z};
        for (int i = 0; i < 3; ++i) {
            NodeMath instruction{static_cast<std::uint32_t>(operation), av[i], bv[i], 0,
                static_cast<StackOffset>(out + i)};
            emitter_.add(NodeType::Math, instruction);
        }
        return stack(out, true);
    }

    ValueRef compileOutput(const mx::NodePtr& node, const std::string& output)
    {
        const ValueRef primary = compile(node);
        if (output.empty())
            return primary;
        const auto outputs = namedOutputs_.find(node.get());
        if (outputs == namedOutputs_.end())
            throw SvmCompileError("SVM: node '" + node->getName()
                + "' does not provide MaterialX output '" + output + "'");
        const auto value = outputs->second.find(output);
        if (value == outputs->second.end())
            throw SvmCompileError("SVM: node '" + node->getName()
                + "' does not provide MaterialX output '" + output + "'");
        return value->second;
    }

    StackOffset allocate(const int count)
    {
        // First-fit allocation mirrors Cycles' SVM stack allocator. Slots are
        // reclaimed once all downstream users have been emitted, keeping
        // large MaterialX graphs within the fixed 255-float device stack.
        for (int offset = 0; offset + count < InvalidOffset; ++offset) {
            bool available = true;
            for (int i = 0; i < count; ++i)
                available &= !stackUsed_[offset + i];
            if (!available)
                continue;
            for (int i = 0; i < count; ++i)
                stackUsed_[offset + i] = true;
            return static_cast<StackOffset>(offset);
        }
        throw SvmCompileError("SVM: stack exhausted (Cycles SVM stack has 255 float slots)");
    }

    void consume(const mx::Node* node)
    {
        const auto use = remainingUses_.find(node);
        if (use == remainingUses_.end() || use->second == 0 || --use->second != 0)
            return;
        const auto value = values_.find(node);
        if (value == values_.end())
            return;
        freeValue(value->second);
        const auto outputs = namedOutputs_.find(node);
        if (outputs != namedOutputs_.end())
            for (const auto& [name, output] : outputs->second)
                if (output.x != value->second.x)
                    freeValue(output);
    }

    void freeValue(const ValueRef value)
    {
        if (!isStackOffset(value.x))
            return;
        const int count = value.vector ? 3 : 1;
        const StackOffset offset = decodeStackOffset(value.x);
        for (int i = 0; i < count; ++i)
            stackUsed_[offset + i] = false;
    }

    static ValueRef stack(const StackOffset offset, const bool vector)
    {
        const std::uint32_t x = encodeStackOffset(offset);
        return {x, encodeStackOffset(static_cast<StackOffset>(offset + (vector ? 1 : 0))),
            encodeStackOffset(static_cast<StackOffset>(offset + (vector ? 2 : 0))), vector};
    }

    static ValueRef coerce(ValueRef value, const bool vector)
    {
        if (vector && !value.vector)
            value.y = value.z = value.x;
        value.vector = vector;
        return value;
    }

    Emitter& emitter_;
    const std::unordered_map<std::string, std::uint32_t>& resolvedTextures_;
    std::unordered_map<const mx::Node*, ValueRef> values_;
    std::unordered_map<const mx::Node*, std::unordered_map<std::string, ValueRef>> namedOutputs_;
    std::unordered_set<const mx::Node*> active_;
    std::unordered_map<const mx::Node*, std::uint32_t> remainingUses_;
    std::vector<std::vector<const mx::Node*>> consumptionScopes_;
    std::unordered_map<std::uint32_t, std::uint32_t> textureSlots_;
    std::vector<std::uint32_t> textures_;
    std::array<bool, StackSize> stackUsed_{};
};

NodeClosureOpenPbrSurface compileOpenPbr(const mx::NodePtr& node, GraphCompiler& graph)
{
    NodeClosureOpenPbrSurface result{};
    const auto scalar = [&](const char* name, const float fallback) {
        return graph.input(node, name, false, fallback).x;
    };
    const auto color = [&](const char* name, const mx::Color3& fallback,
                           std::uint32_t& x, std::uint32_t& y, std::uint32_t& z) {
        const ValueRef value = graph.input(node, name, true, 0.0f, fallback);
        x = value.x;
        y = value.y;
        z = value.z;
    };

    color("base_color", mx::Color3(0.8f), result.baseColorX, result.baseColorY, result.baseColorZ);
    result.baseWeight = scalar("base_weight", 1.0f);
    result.baseDiffuseRoughness = scalar("base_diffuse_roughness", 0.0f);
    result.metalness = scalar("base_metalness", 0.0f);
    result.specularWeight = scalar("specular_weight", 1.0f);
    result.specularRoughness = scalar("specular_roughness", 0.5f);
    result.specularIor = scalar("specular_ior", 1.5f);
    color("specular_color", mx::Color3(1.0f),
        result.specularColorX, result.specularColorY, result.specularColorZ);
    result.transmissionWeight = scalar("transmission_weight", 0.0f);
    color("transmission_color", mx::Color3(1.0f),
        result.transmissionColorX, result.transmissionColorY, result.transmissionColorZ);
    result.subsurfaceWeight = scalar("subsurface_weight", 0.0f);
    color("subsurface_color", mx::Color3(0.8f),
        result.subsurfaceColorX, result.subsurfaceColorY, result.subsurfaceColorZ);
    result.fuzzWeight = scalar("fuzz_weight", 0.0f);
    color("fuzz_color", mx::Color3(1.0f), result.fuzzColorX, result.fuzzColorY, result.fuzzColorZ);
    result.fuzzRoughness = scalar("fuzz_roughness", 0.5f);
    result.coatWeight = scalar("coat_weight", 0.0f);
    color("coat_color", mx::Color3(1.0f), result.coatColorX, result.coatColorY, result.coatColorZ);
    result.coatRoughness = scalar("coat_roughness", 0.0f);
    result.coatIor = scalar("coat_ior", 1.5f);
    if (const mx::InputPtr normal = node->getInput("geometry_normal");
        normal && normal->getConnectedNode()) {
        const ValueRef value = graph.input(node, "geometry_normal", true);
        result.normalX = value.x;
        result.normalY = value.y;
        result.normalZ = value.z;
    }
    else {
        result.normalX = encodeStackOffset(InvalidOffset);
        result.normalY = encodeStackOffset(InvalidOffset);
        result.normalZ = encodeStackOffset(InvalidOffset);
    }
    color("emission_color", mx::Color3(0.0f),
        result.emissionColorX, result.emissionColorY, result.emissionColorZ);
    result.emissionLuminance = scalar("emission_luminance", 0.0f);
    result.opacity = scalar("geometry_opacity", 1.0f);
    return result;
}

// Autodesk standard_surface is the other canonical MaterialX closure.  Keep
// it as a first-class terminal in the SVM front end, but lower it to the same
// lobe payload used by open_pbr_surface.  This is deliberately a data mapping
// rather than a color approximation: the interpreter still performs all
// spectral conversion, Fresnel, transmission and energy compensation in the
// NoorRay BSDFs.  Consequently standard_surface can be freely composed with
// add/mix closure nodes exactly like any other Cycles shader closure.
NodeClosureOpenPbrSurface compileStandardSurface(const mx::NodePtr& node, GraphCompiler& graph)
{
    NodeClosureOpenPbrSurface result{};
    const auto scalar = [&](const char* name, const float fallback) {
        return graph.input(node, name, false, fallback).x;
    };
    const auto color = [&](const char* name, const mx::Color3& fallback,
                           std::uint32_t& x, std::uint32_t& y, std::uint32_t& z) {
        const ValueRef value = graph.input(node, name, true, 0.0f, fallback);
        x = value.x; y = value.y; z = value.z;
    };

    color("base_color", mx::Color3(1.0f), result.baseColorX, result.baseColorY, result.baseColorZ);
    result.baseWeight = scalar("base", 0.8f);
    result.baseDiffuseRoughness = scalar("diffuse_roughness", 0.0f);
    result.metalness = scalar("metalness", 0.0f);
    result.specularWeight = scalar("specular", 1.0f);
    color("specular_color", mx::Color3(1.0f), result.specularColorX, result.specularColorY, result.specularColorZ);
    result.specularRoughness = scalar("specular_roughness", 0.2f);
    result.specularIor = scalar("specular_IOR", 1.5f);
    result.transmissionWeight = scalar("transmission", 0.0f);
    color("transmission_color", mx::Color3(1.0f), result.transmissionColorX,
        result.transmissionColorY, result.transmissionColorZ);
    result.subsurfaceWeight = scalar("subsurface", 0.0f);
    color("subsurface_color", mx::Color3(1.0f), result.subsurfaceColorX,
        result.subsurfaceColorY, result.subsurfaceColorZ);
    result.fuzzWeight = scalar("sheen", 0.0f);
    color("sheen_color", mx::Color3(1.0f), result.fuzzColorX, result.fuzzColorY, result.fuzzColorZ);
    result.fuzzRoughness = scalar("sheen_roughness", 0.3f);
    result.coatWeight = scalar("coat", 0.0f);
    color("coat_color", mx::Color3(1.0f), result.coatColorX, result.coatColorY, result.coatColorZ);
    result.coatRoughness = scalar("coat_roughness", 0.1f);
    result.coatIor = scalar("coat_IOR", 1.5f);
    if (const mx::InputPtr normal = node->getInput("normal");
        normal && normal->getConnectedNode()) {
        const ValueRef value = graph.input(node, "normal", true);
        result.normalX = value.x; result.normalY = value.y; result.normalZ = value.z;
    }
    else {
        result.normalX = encodeStackOffset(InvalidOffset);
        result.normalY = encodeStackOffset(InvalidOffset);
        result.normalZ = encodeStackOffset(InvalidOffset);
    }
    color("emission_color", mx::Color3(1.0f), result.emissionColorX,
        result.emissionColorY, result.emissionColorZ);
    result.emissionLuminance = scalar("emission", 0.0f);
    // Standard Surface opacity is color3.  GraphCompiler's scalar coercion
    // intentionally uses its first channel, matching Cycles' scalar socket
    // behavior for linked RGB values.
    result.opacity = scalar("opacity", 1.0f);
    return result;
}

void emitWeight(Emitter& emitter, const ValueRef value)
{
    ValueRef weight = value;
    if (!weight.vector)
        weight.y = weight.z = weight.x;
    emitter.add(NodeType::ClosureWeight,
        NodeClosureWeight{weight.x, weight.y, weight.z});
}

ValueRef closureWeight(GraphCompiler& graph, const mx::NodePtr& node,
    const ValueRef& inherited)
{
    const ValueRef local = graph.input(node, "weight", false, 1.0f);
    // The nodedef default is one. Do not manufacture three scalar multiply
    // instructions for the overwhelmingly common default leaf case.
    if (!isStackOffset(local.x) && std::bit_cast<float>(local.x) == 1.0f)
        return inherited;
    return graph.multiply(inherited, local);
}

void emitClosureNormal(GraphCompiler& graph, const mx::NodePtr& node,
    std::uint32_t& x, std::uint32_t& y, std::uint32_t& z)
{
    const mx::InputPtr normal = node->getInput("normal");
    if (normal && normal->getConnectedNode()) {
        const ValueRef value = graph.input(node, "normal", true);
        x = value.x;
        y = value.y;
        z = value.z;
    }
    else {
        // defaultgeomprop=Nworld maps to the hit shading normal.  Preserve
        // this sentinel in bytecode rather than spend stack slots per leaf.
        x = y = z = encodeStackOffset(InvalidOffset);
    }
}

void emitClosure(Emitter& emitter, GraphCompiler& graph,
    const mx::NodePtr& node, const ValueRef& inheritedWeight)
{
    const std::string& category = node->getCategory();
    if (category == "surface") {
        if (const mx::NodePtr bsdf = node->getConnectedNode("bsdf"))
            emitClosure(emitter, graph, bsdf, inheritedWeight);
        if (const mx::NodePtr edf = node->getConnectedNode("edf"))
            emitClosure(emitter, graph, edf, inheritedWeight);
        return;
    }
    if (category == "add") {
        if (const mx::NodePtr a = node->getConnectedNode("in1"))
            emitClosure(emitter, graph, a, inheritedWeight);
        if (const mx::NodePtr b = node->getConnectedNode("in2"))
            emitClosure(emitter, graph, b, inheritedWeight);
        return;
    }
    if (category == "layer") {
        // ND_layer_bsdf: vertically layers `top` over `base`.  An exact
        // layered BSDF needs top's directional albedo to attenuate base,
        // which is only known at shade time; approximate it the same way
        // Cycles' own Principled BSDF layers its coat (an unweighted sum of
        // both lobes, each already carrying its own Fresnel falloff) rather
        // than adding new shade-time attenuation machinery.
        if (const mx::NodePtr top = node->getConnectedNode("top"))
            emitClosure(emitter, graph, top, inheritedWeight);
        if (const mx::NodePtr base = node->getConnectedNode("base"))
            emitClosure(emitter, graph, base, inheritedWeight);
        return;
    }
    if (category == "mix") {
        // Cycles' svm_node_mix_closure saturates the factor before deriving
        // the two branch weights.  Do the same here; otherwise out-of-range
        // MaterialX values can produce negative closure energy.
        const ValueRef factor = graph.clamp01(graph.input(node, "mix", false));
        if (const mx::NodePtr background = node->getConnectedNode("bg")) {
            // Cycles' generate_multi_closure emits JUMP_IF_ONE before the
            // first branch, avoiding all branch-local work when the factor is
            // one. The offset is patched after the branch has been emitted.
            const std::size_t jumpStart = emitter.size();
            emitter.add(NodeType::JumpIfOne, NodeJump{factor.x, 0});
            emitClosure(emitter, graph, background,
                graph.multiply(inheritedWeight, graph.oneMinus(factor)));
            emitter.patch(jumpStart + 2,
                static_cast<std::uint32_t>(emitter.size() - (jumpStart + 3)));
        }
        if (const mx::NodePtr foreground = node->getConnectedNode("fg")) {
            const std::size_t jumpStart = emitter.size();
            emitter.add(NodeType::JumpIfZero, NodeJump{factor.x, 0});
            emitClosure(emitter, graph, foreground,
                graph.multiply(inheritedWeight, factor));
            emitter.patch(jumpStart + 2,
                static_cast<std::uint32_t>(emitter.size() - (jumpStart + 3)));
        }
        return;
    }

    if (category == "open_pbr_surface" || category == "standard_surface") {
        // Open-PBR is itself a multi-lobe closure.  Keep the enclosing
        // closure-tree weight in the interpreter's implicit weight register,
        // exactly as Cycles does before evaluating a Principled closure.
        emitWeight(emitter, inheritedWeight);
        if (category == "open_pbr_surface")
            emitter.add(NodeType::ClosureOpenPbrSurface, compileOpenPbr(node, graph));
        else
            emitter.add(NodeType::ClosureOpenPbrSurface, compileStandardSurface(node, graph));
        return;
    }

    const ValueRef weighted = closureWeight(graph, node, inheritedWeight);
    emitWeight(emitter, weighted);
    if (category == "oren_nayar_diffuse_bsdf" || category == "burley_diffuse_bsdf"
        || category == "translucent_bsdf") {
        const ValueRef color = graph.input(node, "color", true, 0.0f, mx::Color3(0.8f));
        NodeClosureDiffuseBsdf instruction{};
        instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
        instruction.roughness = graph.input(node, "roughness", false, 0.0f).x;
        emitClosureNormal(graph, node, instruction.normalX, instruction.normalY, instruction.normalZ);
        instruction.burley = category == "burley_diffuse_bsdf" ? 1u : 0u;
        instruction.translucent = category == "translucent_bsdf" ? 1u : 0u;
        emitter.add(NodeType::ClosureDiffuseBsdf, instruction);
        return;
    }
    if (category == "dielectric_bsdf") {
        const ValueRef tint = graph.input(node, "tint", true, 0.0f, mx::Color3(1.0f));
        NodeClosureDielectricBsdf instruction{};
        instruction.colorX = tint.x; instruction.colorY = tint.y; instruction.colorZ = tint.z;
        instruction.roughness = graph.input(node, "roughness", false, 0.0f).x;
        instruction.ior = graph.input(node, "ior", false, 1.5f).x;
        // scatter_mode is a uniform MaterialX string.  Preserve its exact
        // R versus T/RT branch selection in the compact closure payload;
        // this is not inferred from tint, whose default is white for all
        // three modes.
        const mx::InputPtr scatterMode = node->getInput("scatter_mode");
        const std::string mode = scatterMode
            ? scatterMode->getValueString() : "R";
        instruction.transmission = floatWord(mode == "R" ? 0.0f : 1.0f);
        emitClosureNormal(graph, node, instruction.normalX, instruction.normalY, instruction.normalZ);
        emitter.add(NodeType::ClosureDielectricBsdf, instruction);
        return;
    }
    if (category == "conductor_bsdf") {
        const ValueRef ior = graph.input(node, "ior", true, 0.0f, mx::Color3(1.5f));
        const ValueRef extinction = graph.input(node, "extinction", true, 0.0f, mx::Color3(1.0f));
        NodeClosureConductorBsdf instruction{};
        instruction.colorX = instruction.colorY = instruction.colorZ = floatWord(1.0f);
        instruction.roughness = graph.input(node, "roughness", false, 0.0f).x;
        instruction.anisotropy = floatWord(0.0f);
        instruction.iorX = ior.x; instruction.iorY = ior.y; instruction.iorZ = ior.z;
        instruction.extinctionX = extinction.x; instruction.extinctionY = extinction.y;
        instruction.extinctionZ = extinction.z;
        emitClosureNormal(graph, node, instruction.normalX, instruction.normalY, instruction.normalZ);
        emitter.add(NodeType::ClosureConductorBsdf, instruction);
        return;
    }
    if (category == "sheen_bsdf") {
        const ValueRef color = graph.input(node, "color", true, 0.0f, mx::Color3(1.0f));
        NodeClosureSheenBsdf instruction{};
        instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
        instruction.roughness = graph.input(node, "roughness", false, 0.0f).x;
        emitClosureNormal(graph, node, instruction.normalX, instruction.normalY, instruction.normalZ);
        emitter.add(NodeType::ClosureSheenBsdf, instruction);
        return;
    }
    if (category == "subsurface_bsdf") {
        const ValueRef color = graph.input(node, "color", true, 0.0f, mx::Color3(1.0f));
        NodeClosureSubsurfaceBsdf instruction{};
        instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
        instruction.roughness = floatWord(0.0f);
        emitClosureNormal(graph, node, instruction.normalX, instruction.normalY, instruction.normalZ);
        emitter.add(NodeType::ClosureSubsurfaceBsdf, instruction);
        return;
    }
    if (category == "uniform_edf") {
        const ValueRef color = graph.input(node, "color", true, 0.0f, mx::Color3(0.0f));
        emitter.add(NodeType::ClosureUniformEdf,
            NodeClosureUniformEdf{color.x, color.y, color.z, floatWord(1.0f)});
        return;
    }
    if (category == "conical_edf") {
        // Approximated as a uniform EDF: the cone falloff (inner_angle/
        // outer_angle) needs a shade-time directional term NoorRayCompositeBsdf's
        // EDF lobe does not carry yet, so only the emitted color survives.
        const ValueRef color = graph.input(node, "color", true, 0.0f, mx::Color3(1.0f));
        emitter.add(NodeType::ClosureUniformEdf,
            NodeClosureUniformEdf{color.x, color.y, color.z, floatWord(1.0f)});
        return;
    }
    if (category == "generalized_schlick_edf") {
        // Approximated by tinting the base EDF with color0 (the normal-
        // incidence Schlick term): the grazing-angle color90 term needs a
        // view-dependent evaluation this compile-time EDF tree cannot express.
        const ValueRef color0 = graph.input(node, "color0", true, 0.0f, mx::Color3(1.0f));
        if (const mx::NodePtr base = node->getConnectedNode("base"))
            emitClosure(emitter, graph, base, graph.multiply(inheritedWeight, color0));
        return;
    }
    if (category == "generalized_schlick_bsdf") {
        // Map the Schlick Fresnel curve's normal/grazing reflectivity
        // (color0/color90) through the same reflectivity -> IOR/extinction
        // solve `artistic_ior` uses, then reuse ConductorBsdf/DielectricBsdf
        // exactly as conductor_bsdf/dielectric_bsdf do below.  This loses the
        // curve's `exponent` shaping (fixed at the physical Schlick exponent
        // of 5) and anisotropic roughness (only roughness.x is used).
        const ValueRef color0 = graph.input(node, "color0", true, 0.0f, mx::Color3(1.0f));
        const ValueRef color90 = graph.input(node, "color90", true, 0.0f, mx::Color3(1.0f));
        const mx::InputPtr scatterMode = node->getInput("scatter_mode");
        const std::string mode = scatterMode ? scatterMode->getValueString() : "R";
        const mx::InputPtr roughnessPort = node->getInput("roughness");
        const bool vectorRoughness = roughnessPort && roughnessPort->getType() == "vector2";
        const float roughness = graph.input(node, "roughness", vectorRoughness,
            0.05f, mx::Color3(0.05f)).x;
        const ValueRef weighted = closureWeight(graph, node, inheritedWeight);
        emitWeight(emitter, weighted);
        std::uint32_t normalX, normalY, normalZ;
        emitClosureNormal(graph, node, normalX, normalY, normalZ);
        if (mode == "R") {
            ValueRef extinction{};
            const ValueRef ior = graph.artisticIor(color0, color90, extinction);
            NodeClosureConductorBsdf instruction{};
            instruction.colorX = instruction.colorY = instruction.colorZ = floatWord(1.0f);
            instruction.roughness = floatWord(roughness);
            instruction.anisotropy = floatWord(0.0f);
            instruction.iorX = ior.x; instruction.iorY = ior.y; instruction.iorZ = ior.z;
            instruction.extinctionX = extinction.x; instruction.extinctionY = extinction.y;
            instruction.extinctionZ = extinction.z;
            instruction.normalX = normalX; instruction.normalY = normalY; instruction.normalZ = normalZ;
            emitter.add(NodeType::ClosureConductorBsdf, instruction);
        }
        else {
            NodeClosureDielectricBsdf instruction{};
            instruction.colorX = color0.x; instruction.colorY = color0.y; instruction.colorZ = color0.z;
            instruction.roughness = floatWord(roughness);
            instruction.ior = floatWord(1.5f);
            instruction.transmission = floatWord(1.0f);
            instruction.normalX = normalX; instruction.normalY = normalY; instruction.normalZ = normalZ;
            emitter.add(NodeType::ClosureDielectricBsdf, instruction);
        }
        return;
    }
    throw SvmCompileError("SVM: unsupported MaterialX closure node '" + category
        + "' ('" + node->getName() + "')");
}
} // namespace

CompiledSvmProgram SvmCompiler::compile(const mx::DocumentPtr& document,
    const std::string& elementName,
    const std::unordered_map<std::string, std::uint32_t>& resolvedTextures)
{
    if (!document)
        throw SvmCompileError("SVM: null MaterialX document");

    // Expand any node (UsdPreviewSurface, gltf_pbr, standard_surface_to_*,
    // switch, ramp, ...) whose only MaterialX implementation is a standard-
    // library nodegraph into the primitives below, before searching for a
    // surface terminal or compiling anything.
    flattenNodeGraphs(document);

    mx::NodePtr surface;
    if (!elementName.empty())
        surface = document->getNode(elementName);
    if (!surface) {
        for (const mx::NodePtr& node : document->getNodes()) {
            if (node->getCategory() == "surfacematerial") {
                surface = node->getConnectedNode("surfaceshader");
                if (surface)
                    break;
            }
        }
    }
    if (!surface) {
        for (const mx::NodePtr& node : document->getNodes()) {
            if (node->getCategory() == "open_pbr_surface") {
                surface = node;
                break;
            }
        }
    }
    if (!surface) {
        for (const mx::NodePtr& node : document->getNodes()) {
            if (node->getCategory() == "standard_surface") {
                surface = node;
                break;
            }
        }
    }
    if (!surface)
        throw SvmCompileError("SVM: no surface terminal in MaterialX document");

    Emitter emitter;
    GraphCompiler graph(emitter, resolvedTextures);
    std::unordered_map<const mx::Node*, std::uint32_t> useCounts;
    for (const mx::ElementPtr& element : document->traverseTree()) {
        const mx::NodePtr node = element->asA<mx::Node>();
        if (!node)
            continue;
        for (const mx::InputPtr& input : node->getInputs()) {
            if (const mx::NodePtr upstream = input->getConnectedNode())
                ++useCounts[upstream.get()];
        }
    }
    graph.setRemainingUses(std::move(useCounts));
    emitClosure(emitter, graph, surface,
        ValueRef{floatWord(1.0f), floatWord(1.0f), floatWord(1.0f), true});
    emitter.end();
    return {emitter.take(), graph.textures()};
}
} // namespace nr::svm
