#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "NoorRay/Materials/Graph.h"
#include "NoorRay/Materials/Nodes/NodeAccess.h"
#include "NoorRay/Materials/Nodes/NodeAttributes.h"
#include "NoorRay/Materials/Nodes/NodeDiagnostics.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Materials {

//! Translates the surface of a Blender material into a material graph. Unlinked inputs become parameters, so that materials of one node setup share a graph.
class GraphCompiler {
public:
    struct Result {
        std::shared_ptr<const Materials::Graph> graph;
        std::vector<Core::float4> parameters;
        std::vector<Scene::MaterialTexture> textures;
        bool tangents = false;
        bool vertexColors = false;
        UvStreams streams;
        //! The UV map of the tangents, empty for the render UV map.
        std::string tangentMap;
        //! The Light Falloff of a light shader, which the light applies by distance: radiance scales by distance to this power, and by d^2 / (smooth + d^2).
        std::uint32_t falloffPower = 0;
        float falloffSmooth = 0.0f;
    };

    //! attributeTables gives the UV sets of named attributes, none where no mesh has them; materialIndex is the pass index of the material.
    GraphCompiler(NodeDiagnostics& diagnostics, NodeTextureSource& textures, AttributeTables* attributeTables, std::string materialName,
                  float materialIndex = 0.0f)
        : diagnostics(diagnostics), textures(textures), attributeTables(attributeTables), materialName(std::move(materialName)), materialIndex(materialIndex) {}

    //! The graph of the Surface input; with bumpsDisplacement, the Displacement input bumps every normal without a link, as Cycles' bump method does.
    std::optional<Result> compile(NodeRef output, bool bumpsDisplacement = false);
    //! The graph whose emission is the scalar value of the input identifier of node, for a density that the device bakes.
    std::optional<Result> compileValue(NodeRef node, std::string_view identifier);
    //! The graph whose emission is the object space offset of the Displacement input; it shares the attribute streams of the surface graph.
    std::optional<Result> compileDisplacement(NodeRef output, const Result& surface);
    //! The graph of a light shader, which the device bakes with no receiver distance, so a Light Falloff gives its Strength and the light falls off.
    std::optional<Result> compileLight(NodeRef output);

private:
    using Value = std::uint32_t;

    std::optional<Result> take();

    //! The bump state: the height of a Bump node evaluates at the hit moved by scale times dPdx (axis 1) or dPdy (axis 2).
    struct Bump {
        std::uint8_t axis = 0;
        Value scale = 0;

        auto operator<=>(const Bump&) const = default;
    };

    //! frame is the node group instance that the node is in, 0 for the material tree.
    struct Context {
        std::uint32_t frame = 0;
        Bump bump;
    };

    //! One closure: its weight and the Principled inputs that render it.
    struct Bsdf {
        Value weight = 0;
        Value baseColor = 0;
        Value metallic = 0;
        Value roughness = 0;
        Value ior = 0;
        Value specular = 0;
        Value specularTint = 0;
        Value anisotropic = 0;
        Value rotation = 0;
        Value normal = 0;
        Value tangent = 0;
        Value transmission = 0;
        Value coat = 0;
        Value coatRoughness = 0;
        Value coatIor = 0;
        Value coatTint = 0;
        Value coatNormal = 0;
        Value sheen = 0;
        Value sheenRoughness = 0;
        Value sheenTint = 0;
        Value thinFilmThickness = 0;
        Value thinFilmIor = 0;
        Value mode = 0;
        Value diffuseRoughness = 0;
        Value subsurface = 0;
        Value subsurfaceRadius = 0;
    };

    //! A shader socket: its closures, and the sums of its emission and its transparent colour.
    struct Shader {
        std::vector<Bsdf> bsdfs;
        std::optional<Value> emission;
        std::optional<Value> transparency;
    };

    struct Frame {
        std::uint32_t parent = 0;
        NodeRef group;
    };

    using Key = std::tuple<std::uint32_t, Bump, std::uintptr_t, std::string>;

    SocketRef input(NodeRef node, std::string_view identifier);
    //! The input that a muted node passes to its output.
    SocketRef internalLink(NodeRef node, std::string_view identifier);
    SocketRef defaultInternalLink(NodeRef node, std::string_view identifier);
    std::string socketType(SocketRef socket);
    std::string nodeName(NodeRef node);
    void unsupported(NodeRef node, std::string_view what);

    Value value(const Context& context, NodeRef node, std::string_view identifier);
    Value socketValue(const Context& context, NodeRef node, SocketRef socket);
    Value defaultValue(const Context& context, NodeRef node, SocketRef socket, const std::string& type);
    Value output(const Context& context, NodeRef node, SocketRef socket);
    Value compileOutput(const Context& context, NodeRef node, const std::string& identifier, const std::string& type);
    Value convert(Value value, std::string_view from, std::string_view to);
    Value zero(std::string_view type);
    Value parameter(const std::string& name, const NodeValues& value, std::uint8_t width);

    Value hit(const Context& context, Materials::GraphInput input);
    //! The three rows of a transform from its first input.
    std::array<Value, 3> rows(const Context& context, Materials::GraphInput first);
    Value camera(const Context& context, NodeRef node, const std::string& identifier);
    Value lightPath(const Context& context, NodeRef node, const std::string& identifier);
    Value objectInfo(const Context& context, NodeRef node, const std::string& identifier);
    bool linked(NodeRef node, std::string_view identifier);
    Value ambientOcclusion(const Context& context, NodeRef node, const std::string& identifier);
    Value bevel(const Context& context, NodeRef node);
    Value raycast(const Context& context, NodeRef node, const std::string& identifier);
    Value vectorTransform(const Context& context, NodeRef node);
    Value displacement(const Context& context, NodeRef node);
    Value vectorDisplacement(const Context& context, NodeRef node);
    Value bumpedNormal(Value normal, Value center, Value heightX, Value heightY, Value width, Value distance, Value strength);
    Value shadingNormal(const Context& context);
    void bumpDisplacement(NodeRef output);
    Value lightFalloff(const Context& context, NodeRef node, const std::string& identifier);
    Value uv(const Context& context, std::string_view map);
    //! Two values of a named attribute, 0 where the mesh does not have it.
    Value stream(const Context& context, const std::string& name, UvPart part);
    Value radialTangent(const Context& context, std::string_view axis);
    Value anisotropicTangent(const Context& context, NodeRef node, Value anisotropic);
    Value texCoord(const Context& context, NodeRef node, const std::string& identifier);
    Value geometry(const Context& context, NodeRef node, const std::string& identifier);
    Value math(const Context& context, NodeRef node);
    Value vectorMath(const Context& context, NodeRef node, const std::string& identifier);
    Value mix(const Context& context, NodeRef node, const std::string& identifier);
    Value mixRgb(const Context& context, NodeRef node);
    Value blend(const std::string& mode, Value factor, Value a, Value b);
    Value mapRange(const Context& context, NodeRef node, const std::string& identifier);
    Value colorRamp(const Context& context, NodeRef node, const std::string& identifier);
    Value sampledRamp(NodeRef node, std::span<const float> elements, Value f, const std::string& identifier);
    Value curve(const Context& context, NodeRef node);
    //! samples are the values of the curve at curveSamplePositions(low, high).
    Value sampledCurve(std::span<const float> samples, bool extrapolates, Value x, float low, float high);
    Value separateColor(const Context& context, NodeRef node, const std::string& identifier);
    Value combineColor(const Context& context, NodeRef node);
    Value mapping(const Context& context, NodeRef node);
    Value rotate(Value vector, Value euler, bool inverse);
    Value vectorRotate(const Context& context, NodeRef node);
    Value normalMap(const Context& context, NodeRef node);
    Value bump(const Context& context, NodeRef node);
    Value image(const Context& context, NodeRef node, const std::string& identifier);
    using Coordinate = std::function<Value(const Context&)>;
    //! A texture lookup whose mip level follows coordinate, evaluated again at the hit moved by dPdx and dPdy; with wraps the u change takes the short way.
    Value sample(const Context& context, std::uint32_t slot, const Coordinate& coordinate, bool wraps);
    Value boxTexture(const Context& context, std::uint32_t slot, const Coordinate& vector, float blend);
    Value environment(const Context& context, NodeRef node, const std::string& identifier);
    struct ImageSlot {
        std::uint32_t slot = 0;
        std::uint32_t columns = 1;
        std::uint32_t rows = 1;
    };
    //! None when the node has no image.
    std::optional<ImageSlot> imageSlot(NodeRef node, Scene::TextureWrap wrap);
    Value missingImage(const std::string& identifier);
    Value noise(const Context& context, NodeRef node, const std::string& identifier);
    Value voronoi(const Context& context, NodeRef node, const std::string& identifier);
    Value wave(const Context& context, NodeRef node, const std::string& identifier);
    Value magic(const Context& context, NodeRef node, const std::string& identifier);
    Value checker(const Context& context, NodeRef node, const std::string& identifier);
    Value brick(const Context& context, NodeRef node, const std::string& identifier);
    Value gradient(const Context& context, NodeRef node, const std::string& identifier);
    Value whiteNoise(const Context& context, NodeRef node, const std::string& identifier);
    Value gabor(const Context& context, NodeRef node, const std::string& identifier);
    Value ies(const Context& context, NodeRef node);
    Value sky(const Context& context, NodeRef node);
    Value radialTiling(const Context& context, NodeRef node, const std::string& identifier);
    Value textureVector(const Context& context, NodeRef node);
    Value coordinate(const Context& context, NodeRef node, std::string_view dimensions);

    Shader shaderInput(const Context& context, NodeRef node, std::string_view identifier);
    Shader shader(const Context& context, NodeRef node, SocketRef socket);
    Shader compileShader(const Context& context, NodeRef node);
    Bsdf neutralBsdf(const Context& context, Value weight);
    Value normalInput(const Context& context, NodeRef node, std::string_view identifier);
    Shader principled(const Context& context, NodeRef node, Value weight);
    void warnSubsurfaceMethod(NodeRef node);
    void scale(Shader& shader, Value factor);
    void add(Shader& target, Shader source);
    void finish(Shader shader);

    std::uint32_t frameOf(std::uint32_t parent, NodeRef group);
    std::uint32_t textureSlot(const Scene::MaterialTexture& texture);

    NodeDiagnostics& diagnostics;
    NodeTextureSource& textures;
    AttributeTables* attributeTables;
    std::string materialName;
    float materialIndex;
    //! The graph is the density of a volume, where the Volume Info node reads the grids.
    bool volume = false;
    bool light = false;
    std::optional<Value> displacedNormal;
    Materials::GraphBuilder builder;
    Result result;
    std::vector<Frame> frames{Frame{}};
    std::map<std::pair<std::uint32_t, std::uintptr_t>, std::uint32_t> frameKeys;
    NodeLinks links;
    std::map<Key, Value> values;
    std::map<Key, Shader> shaders;
};

}
