#include "SlangMaterialGenerator.h"

#include <algorithm>
#include <bit>
#include <functional>
#include <optional>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Material.h>
#include <MaterialXGenHw/HwConstants.h>
#include <MaterialXGenHw/Nodes/HwImageNode.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Nodes/SourceCodeNode.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>

#include "Materials/MaterialX/MaterialXDocument.h"
#include "Materials/MaterialX/MaterialXLibraryFiles.h"

namespace mx = MaterialX;

namespace nr::materialx
{

namespace
{
const std::string Target = "noorrayslang";

// Compute the expensive key once, as MaterialX produces source. The runtime
// cache subsequently keys on this compact value, not on a whole source string.
std::uint64_t shaderShape(const std::string_view source)
{
    std::uint64_t result = 14695981039346656037ull;
    for (const unsigned char byte : source) {
        result ^= byte;
        result *= 1099511628211ull;
    }
    return result;
}

// Closure nodes as expressions over MaterialInterface.slang. MaterialX
// substitutes {{input}} with the input's upstream result or parameter.
// Lobes the renderer cannot shade (transmission, sheen, hair, volumes)
// contribute nothing.
const std::pair<const char*, const char*> ClosureImplementations[] = {
    {"ND_oren_nayar_diffuse_bsdf", "mx_oren_nayar_diffuse_bsdf({{weight}}, {{color}}, {{roughness}}, {{normal}}, {{energy_compensation}})"},
    {"ND_burley_diffuse_bsdf", "mx_burley_diffuse_bsdf({{weight}}, {{color}}, {{roughness}}, {{normal}})"},
    {"ND_subsurface_bsdf", "mx_subsurface_bsdf({{weight}}, {{color}}, {{radius}}, {{anisotropy}}, {{normal}})"},
    {"ND_dielectric_bsdf", "mx_dielectric_bsdf({{weight}}, {{tint}}, {{ior}}, {{roughness}}, {{normal}}, {{scatter_mode}})"},
    {"ND_conductor_bsdf", "mx_conductor_bsdf({{weight}}, {{ior}}, {{extinction}}, {{roughness}}, {{normal}})"},
    {"ND_generalized_schlick_bsdf", "mx_generalized_schlick_bsdf({{weight}}, {{color0}}, {{color90}}, {{roughness}}, {{normal}}, {{scatter_mode}})"},
    {"ND_translucent_bsdf", "BSDF()"},
    {"ND_sheen_bsdf", "BSDF()"},
    {"ND_chiang_hair_bsdf", "BSDF()"},
    {"ND_layer_bsdf", "mx_layer_bsdf({{top}}, {{base}})"},
    {"ND_mix_bsdf", "mx_mix_bsdf({{fg}}, {{bg}}, {{mix}})"},
    {"ND_add_bsdf", "addClosures({{in1}}, {{in2}})"},
    {"ND_multiply_bsdfC", "scaleClosure({{in1}}, {{in2}})"},
    {"ND_multiply_bsdfF", "scaleClosure({{in1}}, float3({{in2}}))"},
    {"ND_uniform_edf", "{{color}}"},
    {"ND_conical_edf", "{{color}}"},
    {"ND_measured_edf", "{{color}}"},
    {"ND_generalized_schlick_edf", "{{color0}} * {{base}}"},
    {"ND_mix_edf", "lerp({{bg}}, {{fg}}, {{mix}})"},
    {"ND_add_edf", "{{in1}} + {{in2}}"},
    {"ND_multiply_edfC", "{{in1}} * {{in2}}"},
    {"ND_multiply_edfF", "{{in1}} * {{in2}}"},
    {"ND_absorption_vdf", "VDF()"},
    {"ND_anisotropic_vdf", "VDF()"},
    {"ND_mix_vdf", "VDF()"},
    {"ND_add_vdf", "VDF()"},
    {"ND_multiply_vdfC", "VDF()"},
    {"ND_multiply_vdfF", "VDF()"},
    {"ND_layer_vdf", "{{top}}"},
    {"ND_surface", "mx_surface({{bsdf}}, {{edf}}, {{opacity}})"},
    {"ND_surface_unlit", "mx_surface_unlit({{emission}}, {{emission_color}}, {{opacity}})"},
    {"ND_mix_surfaceshader", "mx_mix_surfaceshader({{fg}}, {{bg}}, {{mix}})"},
    {"ND_convert_float_surfaceshader", "mx_surface_unlit(1.0, float3({{in}}), 1.0)"},
    {"ND_convert_color3_surfaceshader", "mx_surface_unlit(1.0, {{in}}, 1.0)"},
    {"ND_convert_color4_surfaceshader", "mx_surface_unlit(1.0, {{in}}.rgb, 1.0)"},
};

// MaterialX's bundled genslang library intentionally contains only the common
// math and texture implementations. Supply a target implementation for the
// stock tangent-space normal node, rather than inventing a renderer-specific
// material node or rejecting the default ND_normalmap_float nodedef.
const std::pair<const char*, const char*> StandardImplementations[] = {
    {"ND_normalmap_float",
     "normalize(({{in}}.x * 2.0 - 1.0) * {{scale}} * {{tangent}} + "
     "({{in}}.y * 2.0 - 1.0) * {{scale}} * {{bitangent}} + "
     "({{in}}.z * 2.0 - 1.0) * {{normal}})"},
};

mx::DocumentPtr noorRayLibraries()
{
    mx::DocumentPtr libraries = mx::createDocument();
    libraries->copyContentFrom(getSharedStandardLibraries());
    libraries->addTargetDef(Target)->setInheritString(mx::SlangShaderGenerator::TARGET);
    for (const auto& [nodeDef, source] : ClosureImplementations) {
        if (!libraries->getNodeDef(nodeDef))
            throw std::runtime_error(std::string("MaterialX has no node definition ") + nodeDef);
        const mx::ImplementationPtr implementation =
            libraries->addImplementation(std::string("IM_") + nodeDef + "_" + Target);
        implementation->setNodeDefString(nodeDef);
        implementation->setTarget(Target);
        implementation->setAttribute("sourcecode", source);
    }
    for (const auto& [nodeDef, source] : StandardImplementations) {
        if (!libraries->getNodeDef(nodeDef))
            throw std::runtime_error(std::string("MaterialX has no node definition ") + nodeDef);
        const mx::ImplementationPtr implementation =
            libraries->addImplementation(std::string("IM_") + nodeDef + "_" + Target);
        implementation->setNodeDefString(nodeDef);
        implementation->setTarget(Target);
        implementation->setAttribute("sourcecode", source);
    }
    return libraries;
}

// Public uniforms are parameters read from the block, except strings, which
// have no GPU representation and stay constants of the source.
bool isParameter(const mx::ShaderPort& port, const mx::VariableBlock& publicUniforms)
{
    return port.getType() != mx::Type::STRING
        && publicUniforms.find(port.getName()) == &port;
}

// Words a parameter of this type takes in the parameter block.
std::uint32_t parameterWords(const mx::TypeDesc type)
{
    if (type == mx::Type::FLOAT || type == mx::Type::INTEGER || type == mx::Type::BOOLEAN
        || type == mx::Type::FILENAME)
        return 1;
    if (type == mx::Type::VECTOR2)
        return 2;
    if (type == mx::Type::VECTOR3 || type == mx::Type::COLOR3)
        return 3;
    if (type == mx::Type::VECTOR4 || type == mx::Type::COLOR4)
        return 4;
    if (type == mx::Type::MATRIX33)
        return 9;
    if (type == mx::Type::MATRIX44)
        return 16;
    throw std::runtime_error("MaterialX parameter type " + type.getName()
        + " cannot be passed to the realtime renderer");
}

std::string loadParameter(const mx::ShaderPort& port, const std::uint32_t offset)
{
    const mx::TypeDesc type = port.getType();
    const auto word = [offset](const std::uint32_t index) {
        return "parameters[" + std::to_string(offset + index) + "]";
    };
    const std::string& name = port.getVariable();
    if (type == mx::Type::FILENAME)
        return name + ".handle = " + word(0);
    if (type == mx::Type::INTEGER)
        return name + " = asint(" + word(0) + ")";
    if (type == mx::Type::BOOLEAN)
        return name + " = " + word(0) + " != 0u";
    const std::uint32_t count = parameterWords(type);
    if (count == 1)
        return name + " = asfloat(" + word(0) + ")";
    std::string components;
    for (std::uint32_t i = 0; i < count; ++i)
        components += (i ? ", asfloat(" : "asfloat(") + word(i) + ")";
    const std::string constructor = type == mx::Type::MATRIX33 ? "float3x3"
        : type == mx::Type::MATRIX44 ? "float4x4" : "float" + std::to_string(count);
    return name + " = " + constructor + "(" + components + ")";
}

void packParameter(const mx::ShaderPort& port, SlangMaterial& material)
{
    const mx::TypeDesc type = port.getType();
    const std::uint32_t offset = static_cast<std::uint32_t>(material.parameters.size());
    material.parameters.resize(offset + parameterWords(type), 0u);
    if (!port.getValue())
        return;
    // Some ports keep their value in the authored type (an index as a
    // string); read it as the port's type.
    const mx::ValuePtr value = mx::Value::createValueFromStrings(
        port.getValue()->getValueString(), type.getName());
    if (!value)
        throw std::runtime_error("MaterialX parameter " + port.getName()
            + " has no " + type.getName() + " value");
    std::uint32_t* words = material.parameters.data() + offset;
    const auto floats = [words](std::initializer_list<float> values) {
        std::uint32_t i = 0;
        for (const float v : values)
            words[i++] = std::bit_cast<std::uint32_t>(v);
    };
    if (type == mx::Type::FILENAME)
        material.textures.push_back({offset, value->getValueString()});
    else if (type == mx::Type::INTEGER)
        words[0] = std::bit_cast<std::uint32_t>(value->asA<int>());
    else if (type == mx::Type::BOOLEAN)
        words[0] = value->asA<bool>() ? 1u : 0u;
    else if (type == mx::Type::FLOAT)
        floats({value->asA<float>()});
    else if (type == mx::Type::VECTOR2) {
        const auto v = value->asA<mx::Vector2>();
        floats({v[0], v[1]});
    } else if (type == mx::Type::VECTOR3) {
        const auto v = value->asA<mx::Vector3>();
        floats({v[0], v[1], v[2]});
    } else if (type == mx::Type::COLOR3) {
        const auto v = value->asA<mx::Color3>();
        floats({v[0], v[1], v[2]});
    } else if (type == mx::Type::VECTOR4) {
        const auto v = value->asA<mx::Vector4>();
        floats({v[0], v[1], v[2], v[3]});
    } else if (type == mx::Type::COLOR4) {
        const auto v = value->asA<mx::Color4>();
        floats({v[0], v[1], v[2], v[3]});
    } else if (type == mx::Type::MATRIX33) {
        const auto m = value->asA<mx::Matrix33>();
        for (std::size_t i = 0; i < 9; ++i)
            words[i] = std::bit_cast<std::uint32_t>(m[i / 3][i % 3]);
    } else if (type == mx::Type::MATRIX44) {
        const auto m = value->asA<mx::Matrix44>();
        for (std::size_t i = 0; i < 16; ++i)
            words[i] = std::bit_cast<std::uint32_t>(m[i / 4][i % 4]);
    }
}

// The MaterialGeometry member a vertex-data variable is read from.
std::string vertexDataSource(const mx::ShaderPort& port)
{
    const std::string& name = port.getName();
    if (name == mx::HW::T_POSITION_WORLD)
        return "geometry.position";
    if (name == mx::HW::T_NORMAL_WORLD)
        return "geometry.normal";
    if (name == mx::HW::T_TANGENT_WORLD)
        return "geometry.tangent";
    if (name == mx::HW::T_BITANGENT_WORLD)
        return "geometry.bitangent";
    if (name == mx::HW::T_TEXCOORD + "_0")
        return port.getType() == mx::Type::VECTOR3 ? "float3(geometry.uv, 0.0)" : "geometry.uv";
    // `color` is the ordinary MaterialX geompropvalue spelling.  Accept the
    // common aliases too; only this backend translates them to its vertex ABI.
    if (name == mx::HW::T_COLOR + "_0" || name == "color" || name == "Cd"
        || name == "vertex_color" || name == "$inGeomprop_color"
        || name == "$inGeomprop_Cd" || name == "$inGeomprop_vertex_color")
        return port.getType() == mx::Type::COLOR4 ? "geometry.color" : "geometry.color.rgb";
    throw std::runtime_error("MaterialX geometry input " + name
        + " is not available to the realtime renderer");
}

// A constant-folded value, one float per component; empty when the value
// varies (textures, geometry) or comes from a node the folding does not know.
using FoldedValue = std::optional<std::vector<float>>;

FoldedValue valueComponents(const mx::ValuePtr& value)
{
    if (!value)
        return std::nullopt;
    if (value->isA<float>())
        return std::vector{value->asA<float>()};
    if (value->isA<int>())
        return std::vector{static_cast<float>(value->asA<int>())};
    if (value->isA<bool>())
        return std::vector{value->asA<bool>() ? 1.0f : 0.0f};
    const auto components = [](const auto& vector) {
        return std::vector<float>(vector.data(), vector.data() + vector.numElements());
    };
    if (value->isA<mx::Color3>())
        return components(value->asA<mx::Color3>());
    if (value->isA<mx::Color4>())
        return components(value->asA<mx::Color4>());
    if (value->isA<mx::Vector2>())
        return components(value->asA<mx::Vector2>());
    if (value->isA<mx::Vector3>())
        return components(value->asA<mx::Vector3>());
    if (value->isA<mx::Vector4>())
        return components(value->asA<mx::Vector4>());
    return std::nullopt;
}

// Component-wise, broadcasting a single component as MaterialX's FA variants do.
template<class Operation>
FoldedValue foldComponents(const FoldedValue& a, const FoldedValue& b, Operation operation)
{
    if (!a || !b || (a->size() != b->size() && a->size() != 1 && b->size() != 1))
        return std::nullopt;
    std::vector<float> result(std::max(a->size(), b->size()));
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = operation((*a)[a->size() == 1 ? 0 : i], (*b)[b->size() == 1 ? 0 : i]);
    return result;
}

FoldedValue foldInput(const mx::ShaderInput& input);

// Folds the nodes an opacity is typically built from (switches between
// constants, scaling), by the implementation MaterialX chose for them.
FoldedValue foldOutput(const mx::ShaderOutput& output)
{
    const mx::ShaderNode& node = *output.getNode();
    // Graph input sockets hold the material's own parameter values.
    if (node.isAGraph())
        return valueComponents(output.getValue());
    const auto input = [&node](const char* name) -> FoldedValue {
        const mx::ShaderInput* port = node.getInput(name);
        return port ? foldInput(*port) : std::nullopt;
    };
    const std::string& implementation = node.getImplementation().getName();
    // Stdlib nodes are implemented as source (IM_) or as node graphs (NG_).
    const auto is = [&implementation](const std::string& category) {
        return implementation.starts_with("IM_" + category + "_")
            || implementation.starts_with("NG_" + category + "_");
    };
    if (is("constant"))
        return input("value");
    if (is("dot"))
        return input("in");
    if (is("multiply"))
        return foldComponents(input("in1"), input("in2"), std::multiplies<float>{});
    if (is("add"))
        return foldComponents(input("in1"), input("in2"), std::plus<float>{});
    if (is("subtract"))
        return foldComponents(input("in1"), input("in2"), std::minus<float>{});
    if (is("min"))
        return foldComponents(input("in1"), input("in2"),
            [](const float a, const float b) { return std::min(a, b); });
    if (is("max"))
        return foldComponents(input("in1"), input("in2"),
            [](const float a, const float b) { return std::max(a, b); });
    if (is("clamp"))
        return foldComponents(foldComponents(input("in"), input("low"),
            [](const float a, const float b) { return std::max(a, b); }), input("high"),
            [](const float a, const float b) { return std::min(a, b); });
    if (is("mix")) {
        // A constant mix of zero or one selects one side, whatever the other is.
        const FoldedValue mix = input("mix");
        if (mix && std::ranges::all_of(*mix, [](const float m) { return m == 0.0f; }))
            return input("bg");
        if (mix && std::ranges::all_of(*mix, [](const float m) { return m == 1.0f; }))
            return input("fg");
        const FoldedValue bg = input("bg");
        const FoldedValue difference = foldComponents(input("fg"), bg, std::minus<float>{});
        return foldComponents(bg, foldComponents(difference, mix, std::multiplies<float>{}),
            std::plus<float>{});
    }
    if (is("ifgreater")) {
        // A constant comparison selects one side, whatever the other is.
        const FoldedValue value1 = input("value1");
        const FoldedValue value2 = input("value2");
        if (!value1 || !value2 || value1->size() != 1 || value2->size() != 1)
            return std::nullopt;
        return value1->front() > value2->front() ? input("in1") : input("in2");
    }
    if (is("extract")) {
        const FoldedValue value = input("in");
        const FoldedValue index = input("index");
        if (!value || !index || index->empty() || (*index)[0] < 0.0f
            || static_cast<std::size_t>((*index)[0]) >= value->size())
            return std::nullopt;
        return std::vector{(*value)[static_cast<std::size_t>((*index)[0])]};
    }
    if (is("convert")) {
        FoldedValue value = input("in");
        const std::size_t size = output.getType().getSize();
        if (!value || value->empty() || size == 0)
            return std::nullopt;
        // A scalar broadcasts; a vector drops or, like color3 to color4, pads with one.
        if (value->size() == 1)
            return std::vector<float>(size, value->front());
        value->resize(size, 1.0f);
        return value;
    }
    return std::nullopt;
}

FoldedValue foldInput(const mx::ShaderInput& input)
{
    if (const mx::ShaderOutput* connection = input.getConnection())
        return foldOutput(*connection);
    return valueComponents(input.getValue());
}

// Whether the surface's opacity can be below one. A surface node without an
// opacity input is opaque.
bool transparentSurface(const mx::ShaderGraph& graph)
{
    const mx::ShaderOutput* surface = graph.getOutputSocket()->getConnection();
    const mx::ShaderInput* opacity = surface ? surface->getNode()->getInput("opacity") : nullptr;
    if (!opacity)
        return false;
    const FoldedValue value = foldInput(*opacity);
    return !value || std::ranges::any_of(*value, [](const float o) { return o < 1.0f; });
}
} // namespace

// Reads an implementation's source from the embedded MaterialX libraries
// instead of the file system. Its `file` is relative to the library document.
// Base is the node class MaterialX would otherwise use for the implementation.
template <typename Base>
class EmbeddedSourceCodeNode final : public Base
{
public:
    static mx::ShaderNodeImplPtr create() { return std::make_shared<EmbeddedSourceCodeNode>(); }

protected:
    void resolveSourceCode(const mx::InterfaceElement& element, mx::GenContext&) override
    {
        const auto& implementation = static_cast<const mx::Implementation&>(element);
        const std::filesystem::path path = (std::filesystem::path(implementation.getActiveSourceUri())
            .parent_path() / implementation.getAttribute("file")).lexically_normal();
        this->_sourceFilename = mx::FilePath(path.generic_string());
        this->_functionSource = std::string(embeddedLibraryFile(path.generic_string()));
    }
};

class NoorRayShaderGenerator final : public mx::SlangShaderGenerator
{
public:
    NoorRayShaderGenerator()
        : SlangShaderGenerator(mx::TypeSystem::create())
    {
        // The closure types are MaterialInterface.slang's.
        const auto closure = [this](const mx::TypeDesc type, const std::string& name,
                                 const std::string& defaultValue) {
            _syntax->registerTypeSyntax(type, std::make_shared<mx::AggregateTypeSyntax>(
                _syntax.get(), name, defaultValue, mx::EMPTY_STRING));
        };
        closure(mx::Type::BSDF, "BSDF", "BSDF()");
        closure(mx::Type::EDF, "EDF", "EDF(0.0)");
        closure(mx::Type::VDF, "VDF", "VDF()");
        closure(mx::Type::SURFACESHADER, "surfaceshader", "surfaceshader()");

        // SlangShaderGenerator registers HwImageNode for these, which reads
        // its source file from disk; the libraries are embedded instead.
        for (const char* type : {"float", "color3", "color4", "vector2", "vector3", "vector4"})
            registerImplementation(std::string("IM_image_") + type + "_" + mx::SlangShaderGenerator::TARGET,
                EmbeddedSourceCodeNode<mx::HwImageNode>::create);
    }

    const std::string& getTarget() const override { return Target; }

    // Closures are values here, not functions of a lighting context.
    bool nodeNeedsClosureData(const mx::ShaderNode&) const override { return false; }

protected:
    mx::ShaderNodeImplPtr createShaderNodeImplForImplementation(const mx::Implementation&) const override
    {
        return EmbeddedSourceCodeNode<mx::SourceCodeNode>::create();
    }

    // Library includes name a path below "libraries", without the prefix.
    void emitLibraryInclude(const mx::FilePath& filename, mx::GenContext& context,
        mx::ShaderStage& stage) const override
    {
        emitLibraryFile(filename.asString(mx::FilePath::FormatPosix), context, stage);
    }

    // Resolves #include lines from the embedded libraries, relative to the
    // including file, and emits each file once per stage as MaterialX does.
    void emitBlock(const std::string& block, const mx::FilePath& sourceFilename,
        mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        const std::string& include = _syntax->getIncludeStatement();
        const std::string& quote = _syntax->getStringQuote();
        std::istringstream lines(block);
        std::string code;
        for (std::string line; std::getline(lines, line);) {
            const std::size_t open = line.find_first_of(quote);
            const std::size_t close = line.find_last_of(quote);
            if (line.find(include) == std::string::npos || open == std::string::npos
                || close <= open + 1) {
                code += line + '\n';
                continue;
            }
            SlangShaderGenerator::emitBlock(code, sourceFilename, context, stage);
            code.clear();
            std::string name = line.substr(open + 1, close - open - 1);
            mx::tokenSubstitution(getTokenSubstitutions(), name);
            const std::filesystem::path path = (std::filesystem::path(
                sourceFilename.asString(mx::FilePath::FormatPosix)).parent_path() / name).lexically_normal();
            emitLibraryFile(path.generic_string(), context, stage);
        }
        SlangShaderGenerator::emitBlock(code, sourceFilename, context, stage);
    }

    void emitLibraryFile(const std::string& path, mx::GenContext& context,
        mx::ShaderStage& stage) const
    {
        const std::string dependency = "include:" + path;
        if (stage.hasSourceDependency(dependency))
            return;
        stage.addSourceDependency(dependency);
        emitBlock(std::string(embeddedLibraryFile(path)), mx::FilePath(path), context, stage);
    }

    // Materials are evaluated by the ray tracer, not rasterized: there is no
    // vertex stage.
    void emitVertexStage(const mx::ShaderGraph&, mx::GenContext&, mx::ShaderStage&) const override
    {
    }

    // The pixel stage holds the material module: GeneratedMaterial and its
    // hit entry points.
    void emitPixelStage(const mx::ShaderGraph& graph, mx::GenContext& context,
        mx::ShaderStage& stage) const override
    {
        if (!graph.hasClassification(mx::ShaderNode::Classification::SHADER
                | mx::ShaderNode::Classification::SURFACE))
            throw std::runtime_error("MaterialX element " + graph.getName()
                + " is not a surface shader");
        const mx::ShaderGraphOutputSocket* output = graph.getOutputSocket();
        if (!output->getConnection())
            throw std::runtime_error("MaterialX surface shader " + graph.getName()
                + " has no connected output");

        emitLine("import MaterialInterface", stage);
        emitLine("import MaterialHit", stage);
        emitLineBreak(stage);
        emitTypeDefinitions(context, stage);
        emitConstants(context, stage);

        // Uniforms are module statics: public ones are loaded from the
        // parameter block per call, private ones keep their values.
        const mx::VariableBlock& publicUniforms = stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
        for (const auto& [blockName, block] : stage.getUniformBlocks()) {
            if (blockName == mx::HW::LIGHT_DATA)
                continue;
            for (std::size_t i = 0; i < block->size(); ++i) {
                const mx::ShaderPort* port = (*block)[i];
                const bool loaded = isParameter(*port, publicUniforms);
                std::string declaration = "static " + _syntax->getTypeName(port->getType())
                    + " " + port->getVariable();
                if (!loaded && port->getType() != mx::Type::FILENAME) {
                    const std::string value = port->getValue()
                        ? _syntax->getValue(port->getType(), *port->getValue())
                        : _syntax->getDefaultValue(port->getType());
                    if (!value.empty())
                        declaration += " = " + value;
                }
                emitLine(declaration, stage);
            }
        }
        emitLineBreak(stage);

        const mx::VariableBlock& vertexData = stage.getInputBlock(mx::HW::VERTEX_DATA);
        emitLine("struct " + vertexData.getName(), stage, false);
        emitScopeBegin(stage);
        for (std::size_t i = 0; i < vertexData.size(); ++i)
            emitLine(_syntax->getTypeName(vertexData[i]->getType()) + " "
                + vertexData[i]->getVariable(), stage);
        emitScopeEnd(stage, true);
        emitLine("static " + vertexData.getName() + " " + vertexData.getInstance(), stage);
        emitLineBreak(stage);

        emitLibraryInclude("stdlib/genslang/lib/mx_math.slang", context, stage);
        emitLineBreak(stage);
        _tokenSubstitutions[mx::ShaderGenerator::T_FILE_TRANSFORM_UV] = "mx_transform_uv.glsl";
        _tokenSubstitutions[mx::HW::T_TEX_SAMPLER_SIGNATURE] = "SamplerTexture2D tex_sampler";
        emitFunctionDefinitions(graph, context, stage);

        emitLine("struct GeneratedMaterial : IMaterial", stage, false);
        emitScopeBegin(stage);
        emitLine("static surfaceshader evaluate(MaterialGeometry geometry, uint* parameters)",
            stage, false);
        emitScopeBegin(stage);
        std::uint32_t offset = 0;
        for (std::size_t i = 0; i < publicUniforms.size(); ++i) {
            if (!isParameter(*publicUniforms[i], publicUniforms))
                continue;
            emitLine(loadParameter(*publicUniforms[i], offset), stage);
            offset += parameterWords(publicUniforms[i]->getType());
        }
        for (std::size_t i = 0; i < vertexData.size(); ++i)
            emitLine(getVertexDataPrefix(vertexData) + vertexData[i]->getVariable() + " = "
                + vertexDataSource(*vertexData[i]), stage);
        emitFunctionCalls(graph, context, stage);
        emitLine("return " + output->getConnection()->getVariable(), stage);
        emitScopeEnd(stage);
        emitScopeEnd(stage, true);
        emitLineBreak(stage);

        // The hit stages MaterialHit.slang implements, one hit group per material.
        const auto emitStage = [&](const std::string& stageName, const std::string& entryPoint,
                                   const std::string& payload, const std::string& call) {
            emitLine("[shader(\"" + stageName + "\")]", stage, false);
            emitLine("void " + entryPoint + "(inout " + payload
                + " payload, in BuiltInTriangleIntersectionAttributes attributes)", stage, false);
            emitScopeBegin(stage);
            emitLine(call, stage);
            emitScopeEnd(stage);
        };
        emitStage("closesthit", "closestHit", "RealtimeHitPayload",
            "materialClosestHit<GeneratedMaterial>(payload, attributes.barycentrics)");
        emitStage("anyhit", "anyHit", "RealtimeHitPayload",
            "materialAnyHit<GeneratedMaterial>(attributes.barycentrics)");
        emitStage("anyhit", "shadowAnyHit", "RealtimeShadowPayload",
            "materialAnyHit<GeneratedMaterial>(attributes.barycentrics)");
    }
};

SlangMaterialGenerator::SlangMaterialGenerator()
    : libraries_(noorRayLibraries())
    , generator_(std::make_shared<NoorRayShaderGenerator>())
{
}

SlangMaterialGenerator::~SlangMaterialGenerator() = default;

SlangMaterial SlangMaterialGenerator::generate(const mx::DocumentPtr& document) const
{
    const mx::DocumentPtr working = document->copy();
    working->setDataLibrary(libraries_);

    const std::vector<mx::TypedElementPtr> renderable = mx::findRenderableElements(working);
    if (renderable.empty())
        throw std::runtime_error("MaterialX document has no renderable element");
    mx::TypedElementPtr element = renderable.front();
    if (const mx::NodePtr node = element->asA<mx::Node>();
        node && node->getType() == mx::MATERIAL_TYPE_STRING) {
        const std::vector<mx::NodePtr> shaders =
            mx::getShaderNodes(node, mx::SURFACE_SHADER_TYPE_STRING);
        if (shaders.empty())
            throw std::runtime_error("MaterialX material " + node->getName()
                + " has no surface shader");
        element = shaders.front();
    }

    mx::GenContext context(generator_);
    mx::GenOptions& options = context.getOptions();
    options.shaderInterfaceType = mx::SHADER_INTERFACE_COMPLETE;
    options.hwMaxActiveLightSources = 0;
    options.hwSpecularEnvironmentMethod = mx::SPECULAR_ENVIRONMENT_NONE;
    options.hwTransparency = false;
    options.fileTextureVerticalFlip = false;

    const mx::ShaderPtr shader = generator_->generate("Material", element, context);
    const mx::ShaderStage& stage = shader->getStage(mx::Stage::PIXEL);
    SlangMaterial material;
    material.source = stage.getSourceCode();
    material.transparent = transparentSurface(shader->getGraph());
    material.shaderShape = shaderShape(material.source);
    const mx::VariableBlock& publicUniforms = stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
    for (std::size_t i = 0; i < publicUniforms.size(); ++i)
        if (isParameter(*publicUniforms[i], publicUniforms))
            packParameter(*publicUniforms[i], material);
    return material;
}

} // namespace nr::materialx
