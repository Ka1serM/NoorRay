// MaterialX front end for the fixed SVM instruction stream.
#include "Materials/SVM/SvmCompiler.h"

#include <bit>
#include <cstring>
#include <limits>
#include <array>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

#include <glm/vec4.hpp>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>

#include "Materials/MaterialX/MaterialXDocument.h"
#include "Materials/SVM/SvmTypes.h"

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

// MaterialX represents a link either directly on an input (the form emitted
// by the node editor) or through a named output (the form used by nodegraphs
// and several exporters). Closure composition must accept both forms; using
// only getConnectedNode() silently drops a mix branch when the latter form is
// used.
mx::NodePtr connectedNode(const mx::NodePtr& node, const char* inputName)
{
    if (!node)
        return {};
    const mx::InputPtr input = node->getInput(inputName);
    if (!input)
        return {};
    if (const mx::OutputPtr output = input->getConnectedOutput())
        return output->getConnectedNode();
    return input->getConnectedNode();
}

// Native categories are emitted directly. Other standard-library nodegraphs
// are flattened into their primitive nodes before emission.
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
        "combine2", "combine3", "combine4", "separate2", "separate3", "separate4",
        "premult", "unpremult",
        "remap", "smoothstep", "range",
        "hsvadjust", "invert", "contrast", "saturate", "rgbtohsv", "hsvtorgb", "clamp",
        "texcoord", "position", "normal", "tangent", "bitangent", "geomcolor", "viewdirection",
        "geometry_normal", "geometry_coat_normal", "geometry_tangent", "geometry_coat_tangent",
        "geompropvalue",
        "mapping", "rotate2d", "rotate3d", "gradient", "frame", "time",
        "transformpoint", "transformvector", "transformnormal", "transformmatrix",
        "creatematrix", "transpose", "determinant", "invertmatrix",
        "image", "normalmap", "convert", "luminance", "blackbody", "extract",
        "randomfloat",
        "noorray_sellmeier_ior",
        "fractal2d", "fractal3d", "noise2d", "noise3d", "worleynoise2d", "worleynoise3d",
        "cellnoise2d", "cellnoise3d", "unifiednoise2d", "unifiednoise3d",
        "checkerboard", "roughness_anisotropy", "roughness_dual", "artistic_ior",
        // Closure nodes (emitClosure)
        "surface", "layer", "add", "mix", "open_pbr_surface", "standard_surface",
        "surface_unlit", "disney_principled",
        "oren_nayar_diffuse_bsdf", "burley_diffuse_bsdf", "translucent_bsdf",
        "dielectric_bsdf", "conductor_bsdf", "sheen_bsdf", "subsurface_bsdf", "uniform_edf",
        "conical_edf", "generalized_schlick_edf", "generalized_schlick_bsdf",
        // Top-level wrappers SvmCompiler::compile() searches for directly
        "surfacematerial", "volumematerial",
    };
    return categories.contains(category);
}

// Return only nodes that can contribute to a renderable terminal. MaterialX
// documents commonly contain spare nodes while they are being authored; those
// nodes must not be validated, flattened, or counted as live SVM inputs.
std::unordered_set<const mx::Node*> reachableMaterialNodes(
    const mx::DocumentPtr& document)
{
    std::unordered_set<const mx::Node*> reachable;
    std::function<void(const mx::NodePtr&)> visit = [&](const mx::NodePtr& node) {
        if (!node || !reachable.insert(node.get()).second)
            return;
        for (const mx::InputPtr& input : node->getInputs()) {
            if (!input)
                continue;
            if (const mx::OutputPtr output = input->getConnectedOutput())
                visit(output->getConnectedNode());
            visit(input->getConnectedNode());
        }
    };

    for (const mx::NodePtr& node : document->getNodes()) {
        if (!node)
            continue;
        if (node->getCategory() == "surfacematerial") {
            visit(node);
            visit(connectedNode(node, "surfaceshader"));
        }
        else if (node->getCategory() == "open_pbr_surface"
            || node->getCategory() == "standard_surface")
            visit(node);
    }
    return reachable;
}

// Inlines every node whose MaterialX implementation is a standard-library
// <nodegraph> (rather than native source code) into its constituent nodes,
// recursively, using MaterialX's own graph-rewriting utility -- the same
// mechanism a shader generator uses to flatten functional graphs, just
// applied ahead of SvmCompiler's own fixed primitive set instead of generating
// target-specific shader source. Nodes already handled by isNativelySupportedCategory are
// left untouched (the filter returns false for them) so previously-compiled
// materials emit identical bytecode.
bool hasResolvableNodeDefs(const mx::DocumentPtr& document,
    std::string& unresolved)
{
    const mx::ConstDocumentPtr library = document->getDataLibrary();
    const std::string documentUri = document->getSourceUri();
    const std::unordered_set<const mx::Node*> reachable =
        reachableMaterialNodes(document);
    const auto validateGraph = [&](const mx::GraphElementPtr& graph) {
        for (const mx::NodePtr& node : graph->getNodes()) {
            if (!node || !reachable.contains(node.get())
                || isNativelySupportedCategory(node->getCategory()))
                continue;

            // Only inspect nodes owned by the authored document. The attached
            // standard library also appears in MaterialX's tree, but its
            // implementation graphs are definitions used to resolve authored
            // nodes, not authored SVM input themselves.
            const std::string nodeUri = node->getSourceUri();
            if (!nodeUri.empty() && nodeUri != documentUri)
                continue;

            bool found = false;
            if (library) {
                const auto definitions = node->hasNodeDefString()
                    ? std::vector<mx::NodeDefPtr>{
                        library->getNodeDef(node->getNodeDefString())}
                    : library->getMatchingNodeDefs(node->getCategory());
                for (const mx::NodeDefPtr& definition : definitions) {
                    if (definition && definition->getType() == node->getType()) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                unresolved = node->getCategory() + " (" + node->getName() + ")";
                return false;
            }
        }
        return true;
    };

    if (!validateGraph(document))
        return false;

    for (const mx::NodeGraphPtr& graph : document->getNodeGraphs()) {
        // getNodeGraphs() can expose implementation graphs from the attached
        // MaterialX data library. They are immutable definitions, not part of
        // the authored material graph, and must not be validated or rewritten.
        if (graph->getNodeDef())
            continue;
        const std::string graphUri = graph->getSourceUri();
        if (!graphUri.empty() && graphUri != documentUri)
            continue;
        if (!validateGraph(graph))
            return false;
    }
    return true;
}

void flattenNodeGraphs(const mx::DocumentPtr& document)
{
    std::string unresolved;
    if (!hasResolvableNodeDefs(document, unresolved))
        throw SvmCompileError("SVM: unresolved MaterialX nodedef for node " + unresolved);

    // MaterialX's flattenSubgraphs() rewrites child ordering while it expands
    // graph implementations. Calling it for a graph containing only native
    // SVM nodes is unnecessary and, with some authored node ordering, can
    // trigger its internal "Invalid child index" guard. Only enter the
    // rewrite path when this graph actually contains a non-native node.
    const std::unordered_set<const mx::Node*> reachable =
        reachableMaterialNodes(document);
    const auto needsFlatten = [&reachable](const mx::GraphElementPtr& graph) {
        for (const mx::NodePtr& node : graph->getNodes()) {
            if (reachable.contains(node.get())
                && !isNativelySupportedCategory(node->getCategory()))
                return true;
        }
        return false;
    };

    // Use an unknown target so MaterialX selects the target-agnostic
    // implementation when a nodegraph is available.
    const mx::NodePredicate filter = [&reachable](const mx::NodePtr& node) {
        return node && reachable.contains(node.get())
            && !isNativelySupportedCategory(node->getCategory());
    };
    if (needsFlatten(document)) {
        try {
            document->flattenSubgraphs("none", filter);
        } catch (const std::exception& error) {
            throw SvmCompileError(
                "SVM: MaterialX document graph flattening failed: "
                + std::string(error.what()));
        }
    }
    // flattenSubgraphs() only walks its own direct GraphElement (here, the
    // document's top-level nodes) -- a material's own authored <nodegraph>
    // (for example a hand-built normal-map chain referenced via
    // input/nodegraph="...") is a separate GraphElement whose children it
    // never visits, so flatten each one too.
    const std::string documentUri = document->getSourceUri();
    for (const mx::NodeGraphPtr& nodeGraph : document->getNodeGraphs()) {
        // getNodeGraphs() can expose implementation graphs from the attached
        // MaterialX data library. They are immutable definitions, not part of
        // the authored material graph, and must never be rewritten in place.
        const std::string graphUri = nodeGraph->getSourceUri();
        if (nodeGraph->getNodeDef()
            || (!graphUri.empty() && graphUri != documentUri))
            continue;
        if (!needsFlatten(nodeGraph))
            continue;
        try {
            nodeGraph->flattenSubgraphs("none", filter);
        } catch (const std::exception& error) {
            throw SvmCompileError(
                "SVM: MaterialX nodegraph '" + nodeGraph->getName()
                + "' flattening failed: " + error.what());
        }
    }
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
        // Preserve the authored MaterialX value type.
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
    std::uint32_t w{};
    std::uint8_t width{};
    bool matrix{};
};

int valueWidth(const std::string& type)
{
    if (type == "vector2" || type == "color2") return 2;
    if (type == "vector3" || type == "color3") return 3;
    if (type == "vector4" || type == "color4") return 4;
    return 1;
}

TransformSpace parseTransformSpace(const mx::InputPtr& input)
{
    if (!input)
        return TransformSpace::Identity;
    const std::string value = input->getValueString();
    if (value == "world")
        return TransformSpace::World;
    if (value == "object" || value == "model")
        return TransformSpace::Object;
    return TransformSpace::Identity;
}

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
        const auto finishConnectedValue = [this, vector](ValueRef value) {
            value = coerce(value, vector);
            // Closure instructions retain stack offsets in their payload and
            // execute after graph compilation has finished. Do not let the
            // normal node-lifetime reclamation reuse a terminal input slot.
            if (consumptionScopes_.empty())
                retainValue(value);
            return value;
        };
        const mx::InputPtr port = node->getInput(name);
        // Resolve named nodegraph outputs to their upstream node and output.
        if (port) {
            if (const mx::OutputPtr output = port->getConnectedOutput()) {
                if (const mx::NodePtr connected = output->getConnectedNode()) {
                    const ValueRef value = compileOutput(connected, output->getOutputString());
                    if (!consumptionScopes_.empty())
                        consumptionScopes_.back().push_back(connected.get());
                    return finishConnectedValue(value);
                }
            }
        }
        if (port && port->getConnectedNode()) {
            const mx::NodePtr connected = port->getConnectedNode();
            const ValueRef value = compileOutput(connected, port->getOutputString());
            if (!consumptionScopes_.empty())
                consumptionScopes_.back().push_back(connected.get());
            return finishConnectedValue(value);
        }
        if (port && (port->getType() == "matrix33" || port->getType() == "matrix44")) {
            const int dimension = port->getType() == "matrix33" ? 3 : 4;
            const int width = dimension * dimension;
            const StackOffset out = allocate(width);
            NodeMatrixValue instruction{};
            instruction.resultOffset = out;
            instruction.width = static_cast<StackOffset>(width);
            const mx::ValuePtr value = node->getInputValue(name);
            if (value) {
                if (dimension == 3) {
                    const mx::Matrix33 matrix = value->asA<mx::Matrix33>();
                    for (int row = 0; row < 3; ++row)
                        for (int column = 0; column < 3; ++column)
                            instruction.values[row * 3 + column] =
                                floatWord(matrix[row][column]);
                }
                else {
                    const mx::Matrix44 matrix = value->asA<mx::Matrix44>();
                    for (int row = 0; row < 4; ++row)
                        for (int column = 0; column < 4; ++column)
                            instruction.values[row * 4 + column] =
                                floatWord(matrix[row][column]);
                }
            }
            emitter_.add(NodeType::MatrixValue, instruction);
            return matrixStack(out, width);
        }
        // Decode each port using its declared type and apply scalar promotion
        // only at the consuming operation.
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
        if (port && (port->getType() == "color4" || port->getType() == "vector4")) {
            const mx::ValuePtr value = node->getInputValue(name);
            if (value) {
                const glm::vec4 components = port->getType() == "color4"
                    ? glm::vec4(value->asA<mx::Color4>()[0], value->asA<mx::Color4>()[1],
                        value->asA<mx::Color4>()[2], value->asA<mx::Color4>()[3])
                    : glm::vec4(value->asA<mx::Vector4>()[0], value->asA<mx::Vector4>()[1],
                        value->asA<mx::Vector4>()[2], value->asA<mx::Vector4>()[3]);
                return {floatWord(components.x), floatWord(components.y),
                    floatWord(components.z), true, floatWord(components.w), 4};
            }
        }
        const mx::Color3 color = colorInput(node, name, colorFallback);
        return {floatWord(color[0]), floatWord(color[1]), floatWord(color[2]), true,
            0, static_cast<std::uint8_t>(port && (port->getType() == "color2"
                || port->getType() == "vector2") ? 2 : 3)};
    }

    bool sellmeierInput(const mx::NodePtr& node, const char* name,
        NodeSellmeierIor& result) const
    {
        const mx::NodePtr source = connectedNode(node, name);
        if (!source || source->getCategory() != "noorray_sellmeier_ior")
            return false;
        const auto found = sellmeierOutputs_.find(source.get());
        if (found == sellmeierOutputs_.end())
            return false;
        result = found->second;
        return true;
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
        const int width = valueWidth(node->getType());
        const bool vector = width > 1;
        const bool matrix = node->getType() == "matrix33" || node->getType() == "matrix44";
        const bool matrixOperation = category == "add" || category == "subtract"
            || category == "multiply" || category == "divide"
            || category == "transpose" || category == "determinant"
            || category == "invertmatrix" || category == "ifgreater"
            || category == "ifgreatereq" || category == "ifequal";
        if (matrix && category != "constant" && category != "dot"
            && category != "creatematrix" && !matrixOperation)
            throw SvmCompileError("SVM: unsupported MaterialX matrix node '"
                + category + "' ('" + node->getName() + "')");
        ValueRef result{};
        if (matrix && (category == "add" || category == "subtract"
                || category == "multiply" || category == "divide")) {
            const ValueRef a = input(node, "in1", false);
            const ValueRef b = input(node, "in2", false);
            const int matrixWidth = node->getType() == "matrix33" ? 9 : 16;
            const bool rhsScalar = node->getInput("in2")
                && node->getInput("in2")->getType() != "matrix33"
                && node->getInput("in2")->getType() != "matrix44";
            const StackOffset out = allocate(matrixWidth);
            NodeMatrixBinary instruction{};
            instruction.in1 = a.x;
            instruction.in2 = b.x;
            instruction.resultOffset = out;
            instruction.width = static_cast<StackOffset>(matrixWidth);
            instruction.operation = category == "add" ? MatrixBinaryOp::Add
                : category == "subtract" ? MatrixBinaryOp::Subtract
                : category == "multiply" ? MatrixBinaryOp::Multiply : MatrixBinaryOp::Divide;
            instruction.rhsIsScalar = rhsScalar ? 1 : 0;
            emitter_.add(NodeType::MatrixBinary, instruction);
            result = matrixStack(out, matrixWidth);
        }
        else if (category == "transpose" || category == "invertmatrix") {
            const ValueRef value = input(node, "in", false);
            const int matrixWidth = node->getInput("in")->getType() == "matrix33" ? 9 : 16;
            const StackOffset out = allocate(matrixWidth);
            NodeMatrixUnary instruction{};
            instruction.in = value.x;
            instruction.resultOffset = out;
            instruction.width = static_cast<StackOffset>(matrixWidth);
            instruction.operation = category == "transpose"
                ? MatrixUnaryOp::Transpose : MatrixUnaryOp::Inverse;
            emitter_.add(NodeType::MatrixUnary, instruction);
            result = matrixStack(out, matrixWidth);
        }
        else if (category == "determinant") {
            const ValueRef value = input(node, "in", false);
            const int matrixWidth = node->getInput("in")->getType() == "matrix33" ? 9 : 16;
            const StackOffset out = allocate(1);
            emitter_.add(NodeType::MatrixDeterminant,
                NodeMatrixDeterminant{value.x, static_cast<StackOffset>(matrixWidth), out});
            result = stack(out, false);
        }
        else if (matrix && (category == "ifgreater" || category == "ifgreatereq"
                || category == "ifequal")) {
            const StackOffset condition = allocate(1);
            NodeMath compare{};
            compare.mathType = static_cast<std::uint32_t>(category == "ifgreater"
                ? MathOp::GreaterThan : category == "ifgreatereq"
                    ? MathOp::GreaterEqual : MathOp::Equal);
            compare.value1 = input(node, "value1", false).x;
            compare.value2 = input(node, "value2", false).x;
            compare.resultOffset = condition;
            emitter_.add(NodeType::Math, compare);
            const ValueRef falseValue = input(node, "in2", false);
            const ValueRef trueValue = input(node, "in1", false);
            const int matrixWidth = node->getType() == "matrix33" ? 9 : 16;
            const StackOffset out = allocate(matrixWidth);
            emitter_.add(NodeType::MatrixSelect, NodeMatrixSelect{
                encodeStackOffset(condition), trueValue.x, falseValue.x, out,
                static_cast<StackOffset>(matrixWidth)});
            result = matrixStack(out, matrixWidth);
        }
        else if (category == "add" || category == "subtract" || category == "multiply"
            || category == "divide" || category == "min" || category == "max"
            || category == "modulo" || category == "power" || category == "atan2") {
            result = compileBinary(node, width, category);
        }
        else if (category == "constant") {
            // ND_constant_*: passes its `value` input straight through.
            result = input(node, "value", vector, 0.0f, mx::Color3(0.0f));
        }
        else if (category == "noorray_sellmeier_ior") {
            // MaterialX and Blender expose a scalar IOR socket, so retain a
            // green-line compatibility value for ordinary scalar consumers.
            // Dielectric closures also capture the coefficient operands below;
            // NoorRay's spectral evaluator uses those instead of this value.
            const ValueRef b[3]{
                input(node, "b1", false, 1.03961212f),
                input(node, "b2", false, 0.231792344f),
                input(node, "b3", false, 1.01046945f)};
            const ValueRef c[3]{
                input(node, "c1", false, 0.00600069867f),
                input(node, "c2", false, 0.0200179144f),
                input(node, "c3", false, 103.560653f)};
            const std::uint32_t lambdaSquared = floatWord(0.546074f * 0.546074f);
            std::uint32_t nSquared = floatWord(1.0f);
            for (int index = 0; index < 3; ++index) {
                const StackOffset numerator = allocate(1);
                emitter_.add(NodeType::Math, NodeMath{
                    static_cast<std::uint32_t>(MathOp::Multiply),
                    b[index].x, lambdaSquared, 0, numerator});
                const StackOffset denominator = allocate(1);
                emitter_.add(NodeType::Math, NodeMath{
                    static_cast<std::uint32_t>(MathOp::Subtract),
                    lambdaSquared, c[index].x, 0, denominator});
                const StackOffset term = allocate(1);
                emitter_.add(NodeType::Math, NodeMath{
                    static_cast<std::uint32_t>(MathOp::Divide),
                    encodeStackOffset(numerator), encodeStackOffset(denominator), 0, term});
                const StackOffset sum = allocate(1);
                emitter_.add(NodeType::Math, NodeMath{
                    static_cast<std::uint32_t>(MathOp::Add),
                    nSquared, encodeStackOffset(term), 0, sum});
                nSquared = encodeStackOffset(sum);
            }
            const StackOffset clamped = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{
                static_cast<std::uint32_t>(MathOp::Max),
                nSquared, floatWord(1.0f), 0, clamped});
            const StackOffset output = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{
                static_cast<std::uint32_t>(MathOp::Sqrt),
                encodeStackOffset(clamped), 0, 0, output});
            result = stack(output, false);

            NodeSellmeierIor spectral{};
            spectral.enabled = 1;
            spectral.b1 = b[0].x; spectral.b2 = b[1].x; spectral.b3 = b[2].x;
            spectral.c1 = c[0].x; spectral.c2 = c[1].x; spectral.c3 = c[2].x;
            sellmeierOutputs_[node.get()] = spectral;
            for (const ValueRef value : b)
                retainValue(value);
            for (const ValueRef value : c)
                retainValue(value);
        }
        else if (category == "dot") {
            // ND_dot_*: organizational no-op (IM_dot_*_genglsl is a bare
            // "{{in}}" passthrough); useful only for authoring/graph layout.
            result = coerceWidth(input(node, "in", vector), width);
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
            result = compileUnary(node, width, category);
        }
        else if (category == "mix" || category == "plus" || category == "minus"
            || category == "difference" || category == "burn" || category == "dodge"
            || category == "screen" || category == "overlay") {
            const ValueRef a = coerceWidth(input(node, "bg", vector), width);
            const ValueRef b = coerceWidth(input(node, "fg", vector), width);
            const ValueRef f = input(node, "mix", false);
            const StackOffset out = allocate(width);
            NodeMix instruction{};
            instruction.fac = f.x;
            instruction.value1X = a.x; instruction.value1Y = a.y; instruction.value1Z = a.z;
            instruction.value2X = b.x; instruction.value2Y = b.y; instruction.value2Z = b.z;
            instruction.resultOffset = out;
            instruction.isColor = width == 3 ? 1 : 0;
            instruction.blendType = category == "plus" ? MaterialXBlendType::Add
                : category == "minus" ? MaterialXBlendType::Subtract
                : category == "difference" ? MaterialXBlendType::Difference
                : category == "burn" ? MaterialXBlendType::Burn
                : category == "dodge" ? MaterialXBlendType::Dodge
                : category == "screen" ? MaterialXBlendType::Screen
                : category == "overlay" ? MaterialXBlendType::Overlay : MaterialXBlendType::Blend;
            if (width == 3) {
                emitter_.add(NodeType::Mix, instruction);
            }
            else {
                const std::uint32_t aa[4]{a.x, a.y, a.z, a.w};
                const std::uint32_t bb[4]{b.x, b.y, b.z, b.w};
                for (int i = 0; i < width; ++i) {
                    NodeMix component{};
                    component.fac = f.x;
                    component.value1X = aa[i];
                    component.value2X = bb[i];
                    component.resultOffset = static_cast<StackOffset>(out + i);
                    component.blendType = instruction.blendType;
                    emitter_.add(NodeType::Mix, component);
                }
            }
            result = stackWidth(out, width);
        }
        else if (category == "ifgreater" || category == "ifgreatereq" || category == "ifequal") {
            const StackOffset condition = allocate(1);
            NodeMath compare{};
            compare.mathType = static_cast<std::uint32_t>(category == "ifgreater"
                ? MathOp::GreaterThan : category == "ifgreatereq" ? MathOp::GreaterEqual : MathOp::Equal);
            compare.value1 = input(node, "value1", false).x;
            compare.value2 = input(node, "value2", false).x;
            compare.resultOffset = condition;
            emitter_.add(NodeType::Math, compare);

            const ValueRef falseValue = coerceWidth(input(node, "in2", vector), width);
            const ValueRef trueValue = coerceWidth(input(node, "in1", vector), width);
            const StackOffset out = allocate(width);
            NodeMix select{};
            select.fac = encodeStackOffset(condition);
            select.value1X = falseValue.x; select.value1Y = falseValue.y; select.value1Z = falseValue.z;
            select.value2X = trueValue.x; select.value2Y = trueValue.y; select.value2Z = trueValue.z;
            select.resultOffset = out;
            select.isColor = width == 3 ? 1 : 0;
            if (width == 3) {
                emitter_.add(NodeType::Mix, select);
            }
            else {
                const std::uint32_t aa[4]{falseValue.x, falseValue.y, falseValue.z, falseValue.w};
                const std::uint32_t bb[4]{trueValue.x, trueValue.y, trueValue.z, trueValue.w};
                for (int i = 0; i < width; ++i) {
                    NodeMix component{};
                    component.fac = encodeStackOffset(condition);
                    component.value1X = aa[i];
                    component.value2X = bb[i];
                    component.resultOffset = static_cast<StackOffset>(out + i);
                    emitter_.add(NodeType::Mix, component);
                }
            }
            result = stackWidth(out, width);
        }
        else if (category == "magnitude") {
            // ND_magnitude_vector{2,3,4}: the exporter supplies vector3.
            // Keep it in the shared vector instruction family rather than
            // folding it on the host.
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
        else if (category == "combine4") {
            const StackOffset out = allocate(4);
            emitter_.add(NodeType::CombineColor4, NodeCombineColor4{
                input(node, "in1", false).x,
                input(node, "in2", false).x,
                input(node, "in3", false).x,
                input(node, "in4", false).x,
                out});
            result = stackWidth(out, 4);
        }
        else if (category == "combine2" || category == "combine3") {
            const int combineWidth = category == "combine2" ? 2 : 3;
            const StackOffset out = allocate(combineWidth);
            const std::uint32_t x = input(node, "in1", false).x;
            const std::uint32_t y = input(node, "in2", false).x;
            if (category == "combine2") {
                emitter_.add(NodeType::CombineColor2, NodeCombineColor2{x, y, out});
            }
            else {
                NodeCombineColor instruction{};
                instruction.x = x;
                instruction.y = y;
                instruction.z = input(node, "in3", false).x;
                instruction.layout = ColorChannelLayout::Rgb;
                instruction.resultOffset = out;
                emitter_.add(NodeType::CombineColor, instruction);
            }
            result = stackWidth(out, combineWidth);
        }
        else if (category == "separate4") {
            const ValueRef value = input(node, "in", true);
            const StackOffset out = allocate(4);
            emitter_.add(NodeType::SeparateColor4, NodeSeparateColor4{
                value.x, value.y, value.z, value.w,
                out, static_cast<StackOffset>(out + 1),
                static_cast<StackOffset>(out + 2), static_cast<StackOffset>(out + 3)});
            result = stackWidth(out, 4);
            auto& outputs = namedOutputs_[node.get()];
            const mx::InputPtr inputPort = node->getInput("in");
            if (inputPort && inputPort->getType() == "color4") {
                outputs.emplace("outr", stack(out, false));
                outputs.emplace("outg", stack(static_cast<StackOffset>(out + 1), false));
                outputs.emplace("outb", stack(static_cast<StackOffset>(out + 2), false));
                outputs.emplace("outa", stack(static_cast<StackOffset>(out + 3), false));
            }
            else {
                outputs.emplace("outx", stack(out, false));
                outputs.emplace("outy", stack(static_cast<StackOffset>(out + 1), false));
                outputs.emplace("outz", stack(static_cast<StackOffset>(out + 2), false));
                outputs.emplace("outw", stack(static_cast<StackOffset>(out + 3), false));
            }
        }
        else if (category == "premult" || category == "unpremult") {
            const ValueRef value = input(node, "in", true);
            if (value.width < 4)
                throw SvmCompileError("SVM: " + category
                    + " requires a color4/vector4 input ('" + node->getName() + "')");
            const StackOffset out = allocate(4);
            emitter_.add(category == "premult" ? NodeType::Premultiply : NodeType::Unpremultiply,
                NodeColor4Op{value.x, value.y, value.z, value.w, out});
            result = stackWidth(out, 4);
        }
        else if (category == "separate2" || category == "separate3") {
            // The standard library lowers these to one extract per channel.
            // Retain one multi-result separate instruction and expose
            // MaterialX's named output ports from the same stack allocation.
            const ValueRef value = input(node, "in", true);
            const int separateWidth = category == "separate2" ? 2 : 3;
            const StackOffset out = allocate(separateWidth);
            NodeSeparateColor instruction{};
            instruction.colorX = value.x; instruction.colorY = value.y; instruction.colorZ = value.z;
            instruction.layout = ColorChannelLayout::Rgb;
            instruction.resultXOffset = out;
            instruction.resultYOffset = static_cast<StackOffset>(out + 1);
            instruction.resultZOffset = category == "separate3"
                ? static_cast<StackOffset>(out + 2) : InvalidOffset;
            emitter_.add(NodeType::SeparateColor, instruction);
            result = stackWidth(out, separateWidth);
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
            const ValueRef value = coerceWidth(input(node, "in", vector), width);
            const ValueRef inLow = coerceWidth(
                input(node, "inlow", vector, 0.0f, mx::Color3(0.0f)), width);
            const ValueRef inHigh = coerceWidth(
                input(node, "inhigh", vector, 1.0f, mx::Color3(1.0f)), width);
            const ValueRef gamma = coerceWidth(
                input(node, "gamma", vector, 1.0f, mx::Color3(1.0f)), width);
            const ValueRef outLow = coerceWidth(
                input(node, "outlow", vector, 0.0f, mx::Color3(0.0f)), width);
            const ValueRef outHigh = coerceWidth(
                input(node, "outhigh", vector, 1.0f, mx::Color3(1.0f)), width);
            const StackOffset out = allocate(width == 4 ? 4 : vector ? 3 : 1);
            if (width == 4) {
                NodeRange4 instruction{};
                const std::uint32_t values[4]{value.x, value.y, value.z, value.w};
                const std::uint32_t inLows[4]{inLow.x, inLow.y, inLow.z, inLow.w};
                const std::uint32_t inHighs[4]{inHigh.x, inHigh.y, inHigh.z, inHigh.w};
                const std::uint32_t gammas[4]{gamma.x, gamma.y, gamma.z, gamma.w};
                const std::uint32_t outLows[4]{outLow.x, outLow.y, outLow.z, outLow.w};
                const std::uint32_t outHighs[4]{outHigh.x, outHigh.y, outHigh.z, outHigh.w};
                for (int i = 0; i < 4; ++i) {
                    instruction.value[i] = values[i];
                    instruction.inLow[i] = inLows[i];
                    instruction.inHigh[i] = inHighs[i];
                    instruction.gamma[i] = gammas[i];
                    instruction.outLow[i] = outLows[i];
                    instruction.outHigh[i] = outHighs[i];
                }
                instruction.doClamp = input(node, "doclamp", false, 0.0f).x;
                instruction.resultOffset = out;
                emitter_.add(NodeType::Range4, instruction);
                result = stackWidth(out, 4);
            }
            else {
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
                result = stackWidth(out, width);
            }
        }
        else if (category == "hsvadjust") {
            // MaterialX's functional hsvadjust graph maps to a compact
            // NODE_HSV operation. Keep it as one opcode rather than expand
            // rgbtohsv/multiply/add/hsvtorgb into temporary stack values.
            const ValueRef color = input(node, "in", true);
            const ValueRef amount = input(node, "amount", true);
            const StackOffset out = allocate(width == 4 ? 4 : 3);
            NodeHsvAdjust instruction{};
            instruction.hue = amount.x;
            instruction.saturation = amount.y;
            instruction.value = amount.z;
            instruction.fac = floatWord(1.0f);
            instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
            instruction.resultOffset = out;
            emitter_.add(NodeType::HsvAdjust, instruction);
            if (width == 4) {
                const StackOffset resultOffset = allocate(4);
                emitter_.add(NodeType::CombineColor4, NodeCombineColor4{
                    encodeStackOffset(out), encodeStackOffset(static_cast<StackOffset>(out + 1)),
                    encodeStackOffset(static_cast<StackOffset>(out + 2)), color.w, resultOffset});
                result = stackWidth(resultOffset, 4);
            }
            else
                result = stackWidth(out, 3);
        }
        else if (category == "invert") {
            // MaterialX invert is `amount - in`, not a fac-mixed
            // invert. Keep its vector/scalar variants in the existing opcode.
            const ValueRef value = input(node, "in", vector);
            const ValueRef amount = input(node, "amount", vector, 1.0f,
                mx::Color3(1.0f));
            const StackOffset out = allocate(width);
            if (!vector) {
                emitter_.add(NodeType::Invert, NodeInvert{
                    value.x, value.y, value.z, amount.x, amount.y, amount.z, out, 0});
            }
            else {
                const std::uint32_t values[4]{value.x, value.y, value.z, value.w};
                const std::uint32_t amounts[4]{amount.x, amount.y, amount.z, amount.w};
                for (int i = 0; i < width; ++i)
                    emitter_.add(NodeType::Invert, NodeInvert{
                        values[i], values[i], values[i], amounts[i], amounts[i], amounts[i],
                        static_cast<StackOffset>(out + i), 0});
            }
            result = stackWidth(out, width);
        }
        else if (category == "contrast") {
            const ValueRef value = input(node, "in", vector);
            const ValueRef amount = input(node, "amount", vector, 1.0f,
                mx::Color3(1.0f));
            const ValueRef pivot = input(node, "pivot", vector, 0.5f,
                mx::Color3(0.5f));
            const StackOffset out = allocate(width);
            if (!vector) {
                emitter_.add(NodeType::Contrast, NodeContrast{
                    value.x, value.y, value.z, amount.x, amount.y, amount.z,
                    pivot.x, pivot.y, pivot.z, out, 0});
            }
            else {
                const std::uint32_t values[4]{value.x, value.y, value.z, value.w};
                const std::uint32_t amounts[4]{amount.x, amount.y, amount.z, amount.w};
                const std::uint32_t pivots[4]{pivot.x, pivot.y, pivot.z, pivot.w};
                for (int i = 0; i < width; ++i)
                    emitter_.add(NodeType::Contrast, NodeContrast{
                        values[i], values[i], values[i], amounts[i], amounts[i], amounts[i],
                        pivots[i], pivots[i], pivots[i], static_cast<StackOffset>(out + i), 0});
            }
            result = stackWidth(out, width);
        }
        else if (category == "saturate") {
            const ValueRef color = input(node, "in", true);
            const StackOffset out = allocate(width == 4 ? 4 : 3);
            NodeSaturate instruction{};
            instruction.colorX = color.x; instruction.colorY = color.y; instruction.colorZ = color.z;
            instruction.amount = input(node, "amount", false, 1.0f).x;
            const ValueRef luma = input(node, "lumacoeffs", true, 0.0f,
                mx::Color3(0.2722287f, 0.6740818f, 0.0536895f));
            instruction.lumaX = luma.x; instruction.lumaY = luma.y; instruction.lumaZ = luma.z;
            instruction.resultOffset = out;
            emitter_.add(NodeType::Saturate, instruction);
            if (width == 4) {
                const StackOffset resultOffset = allocate(4);
                emitter_.add(NodeType::CombineColor4, NodeCombineColor4{
                    encodeStackOffset(out), encodeStackOffset(static_cast<StackOffset>(out + 1)),
                    encodeStackOffset(static_cast<StackOffset>(out + 2)), color.w, resultOffset});
                result = stackWidth(resultOffset, 4);
            }
            else
                result = stackWidth(out, 3);
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
            if (width == 4) {
                const StackOffset resultOffset = allocate(4);
                emitter_.add(NodeType::CombineColor4, NodeCombineColor4{
                    encodeStackOffset(out), encodeStackOffset(static_cast<StackOffset>(out + 1)),
                    encodeStackOffset(static_cast<StackOffset>(out + 2)), color.w, resultOffset});
                result = stackWidth(resultOffset, 4);
            }
            else
                result = stackWidth(out, 3);
        }
        else if (category == "hsvtorgb") {
            const ValueRef hsv = input(node, "in", true);
            const StackOffset out = allocate(3);
            NodeCombineColor instruction{};
            instruction.x = hsv.x; instruction.y = hsv.y; instruction.z = hsv.z;
            instruction.layout = ColorChannelLayout::Hsv;
            instruction.resultOffset = out;
            emitter_.add(NodeType::CombineColor, instruction);
            if (width == 4) {
                const StackOffset resultOffset = allocate(4);
                emitter_.add(NodeType::CombineColor4, NodeCombineColor4{
                    encodeStackOffset(out), encodeStackOffset(static_cast<StackOffset>(out + 1)),
                    encodeStackOffset(static_cast<StackOffset>(out + 2)), hsv.w, resultOffset});
                result = stackWidth(resultOffset, 4);
            }
            else
                result = stackWidth(out, 3);
        }
        else if (category == "clamp") {
            const ValueRef value = input(node, "in", vector);
            const ValueRef low = input(node, "low", vector, 0.0f, mx::Color3(0.0f));
            const ValueRef high = input(node, "high", vector, 1.0f, mx::Color3(1.0f));
            const StackOffset out = allocate(width);
            const std::uint32_t values[4]{value.x, value.y, value.z, value.w};
            const std::uint32_t lows[4]{low.x, low.y, low.z, low.w};
            const std::uint32_t highs[4]{high.x, high.y, high.z, high.w};
            for (int i = 0; i < width; ++i) {
                NodeClamp instruction{};
                instruction.value = values[i];
                instruction.minValue = lows[i];
                instruction.maxValue = highs[i];
                instruction.resultOffset = static_cast<StackOffset>(out + i);
                emitter_.add(NodeType::Clamp, instruction);
            }
            result = stackWidth(out, width);
        }
        else if (category == "texcoord" || category == "position" || category == "normal"
            || category == "tangent" || category == "bitangent" || category == "geomcolor"
            || category == "viewdirection" || category == "geometry_normal"
            || category == "geometry_coat_normal" || category == "geometry_tangent"
            || category == "geometry_coat_tangent") {
            // Vertex alpha is only reachable through a color4 geomcolor, which
            // needs a fourth stack slot; everything else here is three-wide.
            const bool vertexAlpha =
                category == "geomcolor" && node->getType() == "color4";
            const StackOffset out = allocate(vertexAlpha ? 4 : 3);
            NodeTexCoord instruction{};
            const mx::InputPtr space = node->getInput("space");
            const std::string spaceName = space ? space->getValueString() : "object";
            instruction.source = vertexAlpha ? TexCoordSource::VertexColorAlpha
                : category == "texcoord" ? TexCoordSource::UV
                : category == "position" && spaceName == "world" ? TexCoordSource::World
                : category == "normal" || category == "geometry_normal"
                    || category == "geometry_coat_normal" ? TexCoordSource::Normal
                : category == "tangent" || category == "geometry_tangent"
                    || category == "geometry_coat_tangent" ? TexCoordSource::Tangent
                : category == "bitangent" ? TexCoordSource::Bitangent
                : category == "geomcolor" ? TexCoordSource::VertexColor
                : category == "viewdirection" ? TexCoordSource::ViewDirection
                : TexCoordSource::Object;
            instruction.resultOffset = out;
            emitter_.add(NodeType::TexCoord, instruction);
            result = vertexAlpha ? stackWidth(out, 4) : stack(out, true);
        }
        else if (category == "mapping") {
            const ValueRef value = input(node, "in", true);
            const ValueRef translation = input(node, "translation", true,
                0.0f, mx::Color3(0.0f));
            const ValueRef rotation = input(node, "rotation", true,
                0.0f, mx::Color3(0.0f));
            const ValueRef scale = input(node, "scale", true,
                1.0f, mx::Color3(1.0f));
            const StackOffset out = allocate(3);
            emitter_.add(NodeType::Mapping, NodeMapping{
                value.x, value.y, value.z,
                translation.x, translation.y, translation.z,
                rotation.x, rotation.y, rotation.z,
                scale.x, scale.y, scale.z,
                out});
            result = stack(out, true);
        }
        else if (category == "geompropvalue") {
            const mx::InputPtr propertyPort = node->getInput("geomprop");
            const std::string property = propertyPort ? propertyPort->getValueString() : "";
            const bool outputVector = vector;

            TexCoordSource source = TexCoordSource::Object;
            bool known = true;
            if (property == "UV0" || property == "st" || property == "uv")
                source = TexCoordSource::UV;
            else if (property == "Nworld" || property == "normal")
                source = TexCoordSource::Normal;
            else if (property == "Tworld" || property == "tangent")
                source = TexCoordSource::Tangent;
            else if (property == "Bworld" || property == "bitangent")
                source = TexCoordSource::Bitangent;
            else if (property == "color" || property == "Cd" || property == "vertex_color")
                source = node->getType() == "color4" || node->getType() == "vector4"
                    ? TexCoordSource::VertexColorAlpha : TexCoordSource::VertexColor;
            else if (property == "Pworld" || property == "position" || property == "P")
                source = TexCoordSource::World;
            else if (property == "Pobject")
                source = TexCoordSource::Object;
            else
                known = false;

            if (!known || !outputVector) {
                // MaterialX requires a declared default for missing primvars;
                // using it as the standard fallback rather than inventing a
                // renderer-specific value.
                result = input(node, "default", outputVector, 0.0f, mx::Color3(0.0f));
            }
            else {
                const bool alpha = (node->getType() == "color4" || node->getType() == "vector4")
                    && source == TexCoordSource::VertexColor;
                const StackOffset out = allocate(alpha ? 4 : 3);
                emitter_.add(NodeType::TexCoord,
                    NodeTexCoord{alpha ? TexCoordSource::VertexColorAlpha : source, out});
                result = alpha ? stackWidth(out, 4) : stack(out, true);
            }
        }
        else if (category == "rotate2d") {
            const ValueRef value = input(node, "in", true);
            const StackOffset out = allocate(3);
            emitter_.add(NodeType::Rotate2d, NodeRotate2d{
                value.x, value.y, input(node, "amount", false).x, out});
            result = stack(out, true);
        }
        else if (category == "gradient") {
            const ValueRef position = input(node, "in", true);
            const mx::InputPtr gradientPort = node->getInput("gradient");
            const std::string gradient = gradientPort ? gradientPort->getValueString() : "LINEAR";
            const auto gradientType = gradient == "QUADRATIC" ? GradientType::Quadratic
                : gradient == "EASING" ? GradientType::Easing
                : gradient == "DIAGONAL" ? GradientType::Diagonal
                : gradient == "RADIAL" ? GradientType::Radial
                : gradient == "SPHERICAL" ? GradientType::Spherical
                : gradient == "QUADRATIC_SPHERE" ? GradientType::QuadraticSpherical
                : GradientType::Linear;
            const StackOffset out = allocate(1);
            emitter_.add(NodeType::GradientTexture,
                NodeGradientTexture{position.x, position.y, position.z, gradientType, out});
            result = stack(out, false);
        }
        else if (category == "frame" || category == "time") {
            const StackOffset out = allocate(1);
            emitter_.add(NodeType::Time, NodeTime{
                input(node, "fps", false, 24.0f).x, out,
                static_cast<StackOffset>(category == "frame")});
            result = stack(out, false);
        }
        else if (category == "rotate3d") {
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
        else if (category == "transformpoint" || category == "transformvector"
            || category == "transformnormal") {
            const ValueRef value = coerceWidth(input(node, "in", true), 3);
            const StackOffset out = allocate(3);
            NodeTransform instruction{};
            instruction.inX = value.x;
            instruction.inY = value.y;
            instruction.inZ = value.z;
            instruction.resultOffset = out;
            instruction.kind = category == "transformpoint" ? TransformKind::Point
                : category == "transformnormal" ? TransformKind::Normal : TransformKind::Vector;
            instruction.from = parseTransformSpace(node->getInput("fromspace"));
            instruction.to = parseTransformSpace(node->getInput("tospace"));
            emitter_.add(NodeType::Transform, instruction);
            result = stackWidth(out, 3);
        }
        else if (category == "creatematrix") {
            const bool matrix44 = node->getType() == "matrix44";
            const int dimension = matrix44 ? 4 : 3;
            const int matrixWidth = dimension * dimension;
            const StackOffset out = allocate(matrixWidth);
            NodeMatrixCompose instruction{};
            instruction.resultOffset = out;
            instruction.width = static_cast<StackOffset>(matrixWidth);
            for (int row = 0; row < dimension; ++row) {
                const std::string inputName = "in" + std::to_string(row + 1);
                const mx::InputPtr inputPort = node->getInput(inputName);
                const ValueRef value = coerceWidth(input(node, inputName.c_str(), true), dimension);
                for (int column = 0; column < dimension; ++column)
                    instruction.values[row * dimension + column] =
                        column == 0 ? value.x : column == 1 ? value.y
                        : column == 2 ? value.z
                        // MaterialX's vector3 -> matrix44 implementation
                        // supplies homogeneous row [x,y,z,1], while the
                        // first three rows receive a zero fourth component.
                        : (row == 3 && (!inputPort || inputPort->getType() != "vector4")
                            ? floatWord(1.0f) : value.w);
            }
            emitter_.add(NodeType::MatrixCompose, instruction);
            result = matrixStack(out, matrixWidth);
        }
        else if (category == "transformmatrix") {
            const ValueRef value = input(node, "in", true);
            const ValueRef matrix = input(node, "mat", false);
            if (!matrix.matrix)
                throw SvmCompileError("SVM: transformmatrix requires a matrix input ('"
                    + node->getName() + "')");
            const int inputWidth = valueWidth(node->getInput("in")->getType());
            const int outputWidth = valueWidth(node->getType());
            const StackOffset out = allocate(outputWidth);
            emitter_.add(NodeType::TransformMatrix, NodeTransformMatrix{
                value.x, value.y, value.z, value.w, matrix.x, out,
                static_cast<StackOffset>(inputWidth),
                matrix.width, static_cast<StackOffset>(outputWidth)});
            result = stackWidth(out, outputWidth);
        }
        else if (category == "image") {
            const mx::InputPtr file = node->getInput("file");
            if (!file || file->getValueString().empty())
                throw SvmCompileError("SVM: image node '" + node->getName() + "' has no file");
            const auto texture = resolvedTextures_.find(file->getValueString());
            if (texture == resolvedTextures_.end())
                throw SvmCompileError("SVM: unresolved image '" + file->getValueString() + "'");
            // Keep RGB and alpha contiguous for color4/vector4 values.
            const StackOffset out = allocate(4);
            const StackOffset alpha = static_cast<StackOffset>(out + 3);
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
            const bool outputFour = node->getType() == "color4"
                || node->getType() == "vector4";
            result = outputFour ? stackWidth(out, 4) : stack(out, true);
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
            // Convert changes the port shape; coerce handles scalar promotion.
            result = coerceWidth(input(node, "in", vector), width);
        }
        else if (category == "luminance") {
            // Luminance returns a grayscale value and preserves color4 alpha.
            const ValueRef color = input(node, "in", true);
            const ValueRef coefficients = input(node, "lumacoeffs", true, 0.0f,
                mx::Color3(0.2722287f, 0.6740818f, 0.0536895f));
            const StackOffset gray = allocate(1);
            NodeVectorMath instruction{};
            instruction.mathType = static_cast<std::uint32_t>(VectorMathOp::DotProduct);
            instruction.value1X = color.x; instruction.value1Y = color.y; instruction.value1Z = color.z;
            instruction.value2X = coefficients.x;
            instruction.value2Y = coefficients.y;
            instruction.value2Z = coefficients.z;
            instruction.resultOffset = gray;
            instruction.resultIsScalar = 1;
            emitter_.add(NodeType::VectorMath, instruction);
            const StackOffset out = allocate(width == 4 ? 4 : 3);
            if (width == 4) {
                emitter_.add(NodeType::CombineColor4, NodeCombineColor4{
                    encodeStackOffset(gray), encodeStackOffset(gray), encodeStackOffset(gray),
                    color.w, out});
                result = stackWidth(out, 4);
            }
            else {
                NodeCombineColor grayscale{};
                grayscale.x = encodeStackOffset(gray);
                grayscale.y = encodeStackOffset(gray);
                grayscale.z = encodeStackOffset(gray);
                grayscale.layout = ColorChannelLayout::Rgb;
                grayscale.resultOffset = out;
                emitter_.add(NodeType::CombineColor, grayscale);
                result = stackWidth(out, 3);
            }
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
            const int index = static_cast<int>(floatInput(node, "index", 0.0f));
            if (index == 3) {
                const mx::InputPtr inputPort = node->getInput("in");
                const mx::NodePtr source = inputPort ? inputPort->getConnectedNode() : nullptr;
                const ValueRef in = input(node, "in", true);
                if (in.width >= 4 && isStackOffset(in.w)) {
                    result = stack(static_cast<StackOffset>(decodeStackOffset(in.w)), false);
                }
                else if (in.width >= 4) {
                    result = ValueRef{in.w, in.w, in.w, false};
                }
                else if (source && source->getCategory() == "geomcolor") {
                    // Vertex alpha is the fourth geomcolor component.
                    result = stack(static_cast<StackOffset>(decodeStackOffset(in.x) + 3), false);
                }
                else if (!source || source->getCategory() != "image") {
                    throw SvmCompileError("SVM: extract component 3 requires a MaterialX image or"
                        " geomcolor node ('" + node->getName() + "')");
                }
                else {
                    // Image alpha is the fourth component of the same fetch.
                    result = compileOutput(source, "alpha");
                }
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
        else if (category == "randomfloat") {
            // Directly lower the standard MaterialX randomfloat definition:
            // scale the input, combine it with the integer seed, sample the
            // standard cell-noise primitive, then remap to [min,max].
            const StackOffset scaledInput = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{
                static_cast<std::uint32_t>(MathOp::Multiply),
                input(node, "in", false).x, floatWord(4096.0f), 0, scaledInput});
            const StackOffset coordinates = allocate(2);
            emitter_.add(NodeType::CombineColor2, NodeCombineColor2{
                encodeStackOffset(scaledInput), input(node, "seed", false).x, coordinates});
            const StackOffset noise = allocate(1);
            emitter_.add(NodeType::CellNoiseTexture, NodeCellNoiseTexture{
                encodeStackOffset(coordinates),
                encodeStackOffset(static_cast<StackOffset>(coordinates + 1)),
                floatWord(0.0f), noise, 1});
            const StackOffset range = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{
                static_cast<std::uint32_t>(MathOp::Subtract),
                input(node, "max", false, 1.0f).x,
                input(node, "min", false, 0.0f).x, 0, range});
            const StackOffset scaledNoise = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{
                static_cast<std::uint32_t>(MathOp::Multiply),
                encodeStackOffset(noise), encodeStackOffset(range), 0, scaledNoise});
            const StackOffset out = allocate(1);
            emitter_.add(NodeType::Math, NodeMath{
                static_cast<std::uint32_t>(MathOp::Add),
                input(node, "min", false, 0.0f).x,
                encodeStackOffset(scaledNoise), 0, out});
            result = stack(out, false);
        }
        else if (category == "fractal2d" || category == "fractal3d") {
            // MaterialX stdlib: sum Perlin octaves at progressively higher
            // frequency and lower amplitude.  This is a MaterialX node, not
            // Retain all four standard MaterialX noise inputs.
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
            // Keep multi-result texture nodes compact.
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
    ValueRef compileUnary(const mx::NodePtr& node, const int width, const std::string& category)
    {
        const bool vector = width > 1;
        const ValueRef inputValue = coerceWidth(input(node, "in", vector), width);
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
        // MaterialX emits one scalar operation per component for these
        // vector categories; retaining that representation keeps stack use
        // and literal encoding identical to the scalar path.
        const StackOffset out = allocate(width);
        const std::uint32_t inputs[4]{inputValue.x, inputValue.y, inputValue.z, inputValue.w};
        for (int i = 0; i < width; ++i) {
            NodeMath instruction{static_cast<std::uint32_t>(operation), inputs[i], 0, 0,
                static_cast<StackOffset>(out + i)};
            emitter_.add(NodeType::Math, instruction);
        }
        return stackWidth(out, width);
    }

    ValueRef compileBinary(const mx::NodePtr& node, const int width, const std::string& category)
    {
        const bool vector = width > 1;
        const ValueRef a = coerceWidth(input(node, "in1", vector), width);
        const ValueRef b = coerceWidth(input(node, "in2", vector), width);
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
        const StackOffset out = allocate(width);
        const std::uint32_t av[4]{a.x, a.y, a.z, a.w};
        const std::uint32_t bv[4]{b.x, b.y, b.z, b.w};
        for (int i = 0; i < width; ++i) {
            NodeMath instruction{static_cast<std::uint32_t>(operation), av[i], bv[i], 0,
                static_cast<StackOffset>(out + i)};
            emitter_.add(NodeType::Math, instruction);
        }
        return stackWidth(out, width);
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
        // First-fit allocation keeps stack reuse deterministic. Slots are
        // reclaimed once all downstream users have been emitted, keeping
        // large MaterialX graphs within the fixed 255-float device stack.
        // InvalidOffset is a sentinel, not a reserved stack slot: offsets
        // [0, StackSize) are valid stack offsets. The
        // old strict inequality permanently lost the last float slot and
        // made otherwise-valid large graphs fail early.
        for (int offset = 0; offset + count <= InvalidOffset; ++offset) {
            bool available = true;
            for (int i = 0; i < count; ++i)
                available &= !stackUsed_[offset + i];
            if (!available)
                continue;
            for (int i = 0; i < count; ++i)
                stackUsed_[offset + i] = true;
            return static_cast<StackOffset>(offset);
        }
        throw SvmCompileError("SVM: stack exhausted (SVM stack has 255 float slots)");
    }

    void consume(const mx::Node* node)
    {
        const auto use = remainingUses_.find(node);
        if (use == remainingUses_.end() || use->second == 0 || --use->second != 0)
            return;
        const auto value = values_.find(node);
        if (value == values_.end())
            return;
        // Some MaterialX operations only change the declared port type.  The
        // current SVM representation can preserve the exact same stack
        // storage for those operations (for example color4 -> color3).
        // Keeping the source and result as separate ValueRefs is fine, but
        // freeing the source here would make the result's live stack slots
        // available for reuse.  That is especially visible when a clamp adds
        // one more temporary to a Mix graph.
        if (!hasLiveAlias(node, value->second))
            freeValue(value->second);
        const auto outputs = namedOutputs_.find(node);
        if (outputs != namedOutputs_.end())
            for (const auto& [name, output] : outputs->second)
                if (output.x != value->second.x && !hasLiveAlias(node, output))
                    freeValue(output);
    }

    bool hasLiveAlias(const mx::Node* owner, const ValueRef value) const
    {
        if (!isStackOffset(value.x))
            return false;
        for (const auto& [node, other] : values_) {
            if (node == owner || other.x != value.x || !isStackOffset(other.x))
                continue;
            const auto uses = remainingUses_.find(node);
            if (uses != remainingUses_.end() && uses->second != 0)
                return true;
        }
        return false;
    }

    void freeValue(const ValueRef value)
    {
        if (!isStackOffset(value.x))
            return;
        const int count = value.width != 0 ? value.width : value.vector ? 3 : 1;
        const StackOffset offset = decodeStackOffset(value.x);
        for (int i = 0; i < count; ++i)
            if (!stackPinned_[offset + i])
                stackUsed_[offset + i] = false;
    }

    void retainValue(const ValueRef value)
    {
        if (!isStackOffset(value.x))
            return;
        const int count = value.width != 0 ? value.width : value.vector ? 3 : 1;
        const StackOffset offset = decodeStackOffset(value.x);
        for (int i = 0; i < count; ++i) {
            stackUsed_[offset + i] = true;
            stackPinned_[offset + i] = true;
        }
    }

    static ValueRef stack(const StackOffset offset, const bool vector)
    {
        return stackWidth(offset, vector ? 3 : 1);
    }

    static ValueRef stackWidth(const StackOffset offset, const int width)
    {
        const std::uint32_t x = encodeStackOffset(offset);
        const auto component = [offset, width](const int index) {
            return index < width
                ? encodeStackOffset(static_cast<StackOffset>(offset + index))
                : floatWord(0.0f);
        };
        ValueRef result{ x, component(1), component(2), width > 1,
            component(3), static_cast<std::uint8_t>(width) };
        return result;
    }

    ValueRef coerce(ValueRef value, const bool vector)
    {
        if (value.matrix)
            return value;
        if (vector && !value.vector) {
            // A linked scalar used as a color/vector must become an actual
            // three-slot value at a terminal boundary. Repeating the source
            // stack offset in x/y/z looks equivalent while reading, but it
            // aliases the scalar's lifetime and can be invalidated when the
            // source value is reclaimed or reused by a later graph node.
            // Immediate scalar literals do not need an instruction: repeating
            // their encoded word is already safe.
            if (isStackOffset(value.x) && consumptionScopes_.empty()) {
                const StackOffset out = allocate(3);
                emitter_.add(NodeType::CombineColor, NodeCombineColor{
                    value.x, value.x, value.x, ColorChannelLayout::Rgb, out});
                return stackWidth(out, 3);
            }
            value.y = value.z = value.x;
            value.w = value.x;
            value.width = 3;
        }
        else if (!vector) {
            value.width = 1;
        }
        value.vector = vector;
        return value;
    }

    ValueRef coerceWidth(ValueRef value, const int width)
    {
        if (value.matrix || width <= 1)
            return width <= 1 ? coerce(value, false) : value;
        if (!value.vector) {
            value.y = value.z = value.w = value.x;
            value.width = static_cast<std::uint8_t>(width);
            value.vector = true;
            return value;
        }
        // MaterialX arithmetic inputs normally have the same declared width.
        // For legal scalar/vector promotion, preserve available components and
        // use zero for a missing component rather than reading an unrelated
        // stack slot.
        const std::uint32_t zero = floatWord(0.0f);
        if (value.width < 2) value.y = zero;
        if (value.width < 3) value.z = zero;
        if (value.width < 4) value.w = zero;
        value.width = static_cast<std::uint8_t>(width);
        value.vector = true;
        return value;
    }

    static ValueRef matrixStack(const StackOffset offset, const int width)
    {
        ValueRef result{};
        result.x = encodeStackOffset(offset);
        result.width = static_cast<std::uint8_t>(width);
        result.matrix = true;
        return result;
    }

    Emitter& emitter_;
    const std::unordered_map<std::string, std::uint32_t>& resolvedTextures_;
    std::unordered_map<const mx::Node*, ValueRef> values_;
    std::unordered_map<const mx::Node*, std::unordered_map<std::string, ValueRef>> namedOutputs_;
    std::unordered_map<const mx::Node*, NodeSellmeierIor> sellmeierOutputs_;
    std::unordered_set<const mx::Node*> active_;
    std::unordered_map<const mx::Node*, std::uint32_t> remainingUses_;
    std::vector<std::vector<const mx::Node*>> consumptionScopes_;
    std::unordered_map<std::uint32_t, std::uint32_t> textureSlots_;
    std::vector<std::uint32_t> textures_;
    std::array<bool, StackSize> stackUsed_{};
    std::array<bool, StackSize> stackPinned_{};
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
    graph.sellmeierInput(node, "specular_ior", result.specularSellmeier);
    graph.sellmeierInput(node, "coat_ior", result.coatSellmeier);
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
// add/mix closure nodes exactly like any other MaterialX shader closure.
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
    graph.sellmeierInput(node, "specular_IOR", result.specularSellmeier);
    graph.sellmeierInput(node, "coat_IOR", result.coatSellmeier);
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
    // intentionally uses its first channel for scalar MaterialX inputs
    // behavior for linked RGB values.
    result.opacity = scalar("opacity", 1.0f);
    return result;
}

// MaterialX's built-in Disney nodegraph intentionally uses a perfectly smooth
// transmission lobe. NoorRay keeps Disney as a regular MaterialX node, but
// lowers the node itself so its authored roughness also controls metallic and
// specTrans closures instead of being lost during nodegraph flattening.
NodeClosureOpenPbrSurface compileDisneyPrincipled(const mx::NodePtr& node,
    GraphCompiler& graph)
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

    color("baseColor", mx::Color3(0.16f),
        result.baseColorX, result.baseColorY, result.baseColorZ);
    result.baseWeight = floatWord(1.0f);
    result.baseDiffuseRoughness = scalar("roughness", 0.5f);
    result.metalness = scalar("metallic", 0.0f);
    result.specularWeight = scalar("specular", 0.5f);
    result.specularRoughness = scalar("roughness", 0.5f);
    result.specularIor = scalar("ior", 1.5f);
    graph.sellmeierInput(node, "ior", result.specularSellmeier);
    result.specularColorX = result.specularColorY = result.specularColorZ =
        floatWord(1.0f);
    result.transmissionWeight = scalar("specTrans", 0.0f);
    color("baseColor", mx::Color3(1.0f),
        result.transmissionColorX, result.transmissionColorY,
        result.transmissionColorZ);
    result.subsurfaceWeight = scalar("subsurface", 0.0f);
    color("baseColor", mx::Color3(1.0f),
        result.subsurfaceColorX, result.subsurfaceColorY,
        result.subsurfaceColorZ);
    result.fuzzWeight = scalar("sheen", 0.0f);
    color("baseColor", mx::Color3(1.0f),
        result.fuzzColorX, result.fuzzColorY, result.fuzzColorZ);
    result.fuzzRoughness = scalar("roughness", 0.5f);
    result.coatWeight = scalar("clearcoat", 0.0f);
    result.coatColorX = result.coatColorY = result.coatColorZ =
        floatWord(1.0f);
    result.coatRoughness = graph.oneMinus(
        graph.input(node, "clearcoatGloss", false, 1.0f)).x;
    result.coatIor = floatWord(1.5f);
    result.normalX = result.normalY = result.normalZ =
        encodeStackOffset(InvalidOffset);
    result.emissionColorX = result.emissionColorY = result.emissionColorZ =
        floatWord(0.0f);
    result.emissionLuminance = floatWord(0.0f);
    result.opacity = floatWord(1.0f);
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
    if (normal && (normal->getConnectedNode() || normal->getConnectedOutput())) {
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
    if (!node)
        return;

    const std::string& category = node->getCategory();
    if (category == "surface") {
        if (const mx::NodePtr bsdf = connectedNode(node, "bsdf"))
            emitClosure(emitter, graph, bsdf, inheritedWeight);
        if (const mx::NodePtr edf = connectedNode(node, "edf"))
            emitClosure(emitter, graph, edf, inheritedWeight);
        emitter.add(NodeType::SurfaceOutput,
            NodeSurfaceOutput{graph.input(node, "opacity", false, 1.0f).x});
        return;
    }
    if (category == "surface_unlit") {
        const ValueRef color = graph.input(node, "emission_color", true,
            1.0f, mx::Color3(1.0f));
        const ValueRef strength = graph.input(node, "emission", false, 1.0f);
        emitWeight(emitter, inheritedWeight);
        emitter.add(NodeType::ClosureUniformEdf,
            NodeClosureUniformEdf{color.x, color.y, color.z, strength.x});
        emitter.add(NodeType::SurfaceOutput,
            NodeSurfaceOutput{graph.input(node, "opacity", false, 1.0f).x});
        return;
    }
    if (category == "add") {
        if (const mx::NodePtr a = connectedNode(node, "in1"))
            emitClosure(emitter, graph, a, inheritedWeight);
        if (const mx::NodePtr b = connectedNode(node, "in2"))
            emitClosure(emitter, graph, b, inheritedWeight);
        return;
    }
    if (category == "layer") {
        // ND_layer_bsdf: vertically layers `top` over `base`.  An exact
        // layered BSDF needs top's directional albedo to attenuate base,
        // which is only known at shade time; approximate it the same way
        // The OpenPBR surface layers its coat (an unweighted sum of
        // both lobes, each already carrying its own Fresnel falloff) rather
        // than adding new shade-time attenuation machinery.
        if (const mx::NodePtr top = connectedNode(node, "top"))
            emitClosure(emitter, graph, top, inheritedWeight);
        if (const mx::NodePtr base = connectedNode(node, "base"))
            emitClosure(emitter, graph, base, inheritedWeight);
        return;
    }
    if (category == "mix") {
        // The MaterialX closure mix saturates the factor before deriving
        // the two branch weights.  Do the same here; otherwise out-of-range
        // MaterialX values can produce negative closure energy.
        const ValueRef factor = graph.clamp01(graph.input(node, "mix", false));
        if (const mx::NodePtr background = connectedNode(node, "bg")) {
            // The closure tree emits JUMP_IF_ONE before the
            // first branch, avoiding all branch-local work when the factor is
            // one. The offset is patched after the branch has been emitted.
            const std::size_t jumpStart = emitter.size();
            emitter.add(NodeType::JumpIfOne, NodeJump{factor.x, 0});
            emitClosure(emitter, graph, background,
                graph.multiply(inheritedWeight, graph.oneMinus(factor)));
            emitter.patch(jumpStart + 2,
                static_cast<std::uint32_t>(emitter.size() - (jumpStart + 3)));
        }
        if (const mx::NodePtr foreground = connectedNode(node, "fg")) {
            const std::size_t jumpStart = emitter.size();
            emitter.add(NodeType::JumpIfZero, NodeJump{factor.x, 0});
            emitClosure(emitter, graph, foreground,
                graph.multiply(inheritedWeight, factor));
            emitter.patch(jumpStart + 2,
                static_cast<std::uint32_t>(emitter.size() - (jumpStart + 3)));
        }
        return;
    }

    if (category == "multiply") {
        // MaterialX uses the same category for numeric multiplication and
        // closure scaling.  Once we are walking a closure-typed node, in1 is
        // the closure and in2 is its scalar/color contribution weight.
        if (const mx::NodePtr inputClosure = connectedNode(node, "in1")) {
            const mx::InputPtr scalePort = node->getInput("in2");
            const bool vectorScale = scalePort
                && (scalePort->getType() == "color3" || scalePort->getType() == "vector3");
            emitClosure(emitter, graph, inputClosure,
                graph.multiply(inheritedWeight, graph.input(node, "in2", vectorScale,
                    1.0f, mx::Color3(1.0f))));
        }
        return;
    }

    if (category == "open_pbr_surface" || category == "standard_surface") {
        // Open-PBR is itself a multi-lobe closure.  Keep the enclosing
        // closure-tree weight in the interpreter's implicit weight register,
        // before evaluating the surface closure.
        emitWeight(emitter, inheritedWeight);
        if (category == "open_pbr_surface")
            emitter.add(NodeType::ClosureOpenPbrSurface, compileOpenPbr(node, graph));
        else
            emitter.add(NodeType::ClosureOpenPbrSurface, compileStandardSurface(node, graph));
        return;
    }

    if (category == "disney_principled") {
        emitWeight(emitter, inheritedWeight);
        emitter.add(NodeType::ClosureOpenPbrSurface,
            compileDisneyPrincipled(node, graph));
        emitter.add(NodeType::SurfaceOutput,
            NodeSurfaceOutput{floatWord(1.0f)});
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
        graph.sellmeierInput(node, "ior", instruction.sellmeier);
        // scatter_mode is a uniform MaterialX string. Preserve all three
        // modes in the compact closure payload: R=0, T=1, RT=2. The SVM
        // evaluator uses this to disable the unwanted lobe, rather than
        // treating T and RT as the same kind of glass.
        const mx::InputPtr scatterMode = node->getInput("scatter_mode");
        const std::string mode = scatterMode
            ? scatterMode->getValueString() : "R";
        instruction.transmission = floatWord(
            mode == "R" ? 0.0f : mode == "T" ? 1.0f : 2.0f);
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
        emitWeight(emitter, closureWeight(graph, node, inheritedWeight));
        emitter.add(NodeType::ClosureUniformEdf,
            NodeClosureUniformEdf{color.x, color.y, color.z, floatWord(1.0f)});
        return;
    }
    if (category == "conical_edf") {
        // Approximated as a uniform EDF: the cone falloff (inner_angle/
        // outer_angle) needs a shade-time directional term NoorRayCompositeBsdf's
        // EDF lobe does not carry yet, so only the emitted color survives.
        const ValueRef color = graph.input(node, "color", true, 0.0f, mx::Color3(1.0f));
        emitWeight(emitter, inheritedWeight);
        emitter.add(NodeType::ClosureUniformEdf,
            NodeClosureUniformEdf{color.x, color.y, color.z, floatWord(1.0f)});
        return;
    }
    if (category == "generalized_schlick_edf") {
        // Approximated by tinting the base EDF with color0 (the normal-
        // incidence Schlick term): the grazing-angle color90 term needs a
        // view-dependent evaluation this compile-time EDF tree cannot express.
        const ValueRef color0 = graph.input(node, "color0", true, 0.0f, mx::Color3(1.0f));
        if (const mx::NodePtr base = connectedNode(node, "base"))
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
        // Keep the encoded SVM input word intact.  For the Disney graph this
        // is usually a stack reference produced by roughness_anisotropy;
        // converting it to float and passing it through floatWord() turns the
        // reference into a literal, so metallic and transmission roughness
        // silently stop following the authored roughness.
        const std::uint32_t roughness = graph.input(node, "roughness",
            vectorRoughness, 0.05f, mx::Color3(0.05f)).x;
        const ValueRef weighted = closureWeight(graph, node, inheritedWeight);
        emitWeight(emitter, weighted);
        std::uint32_t normalX, normalY, normalZ;
        emitClosureNormal(graph, node, normalX, normalY, normalZ);
        if (mode == "R") {
            ValueRef extinction{};
            const ValueRef ior = graph.artisticIor(color0, color90, extinction);
            NodeClosureConductorBsdf instruction{};
            instruction.colorX = instruction.colorY = instruction.colorZ = floatWord(1.0f);
            instruction.roughness = roughness;
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
            instruction.roughness = roughness;
            instruction.ior = floatWord(1.5f);
            instruction.transmission = floatWord(mode == "T" ? 1.0f : 2.0f);
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

    // A MaterialX XML round-trip cannot preserve the document's in-memory
    // data-library pointer. Make library availability a compiler invariant so
    // every caller gets the same behavior, including documents loaded from
    // USD and direct compiler users.
    if (!document->getDataLibrary())
        document->setDataLibrary(nr::materialx::getSharedStandardLibraries());

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
                surface = connectedNode(node, "surfaceshader");
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
    const std::unordered_set<const mx::Node*> reachable =
        reachableMaterialNodes(document);
    const auto countUses = [&useCounts, &reachable](const mx::GraphElementPtr& graph) {
        for (const mx::NodePtr& node : graph->getNodes()) {
            if (!node || !reachable.contains(node.get()))
                continue;
            for (const mx::InputPtr& input : node->getInputs()) {
                if (const mx::NodePtr upstream = input->getConnectedNode())
                    ++useCounts[upstream.get()];
                if (const mx::OutputPtr output = input->getConnectedOutput())
                    if (const mx::NodePtr upstream = output->getConnectedNode())
                        ++useCounts[upstream.get()];
            }
        }
    };
    countUses(document);
    for (const mx::NodeGraphPtr& graphNode : document->getNodeGraphs())
        countUses(graphNode);
    graph.setRemainingUses(std::move(useCounts));
    emitClosure(emitter, graph, surface,
        ValueRef{floatWord(1.0f), floatWord(1.0f), floatWord(1.0f), true});
    emitter.end();
    return {emitter.take(), graph.textures()};
}
} // namespace nr::svm
