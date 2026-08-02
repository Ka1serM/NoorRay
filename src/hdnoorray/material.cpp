#include "material.h"

#include "renderParam.h"

#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/imaging/hd/sceneDelegate.h>

#include "Backend/OptiX/Runtime/Raytracer.h"

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>

#include <Materials/MaterialX/MaterialXDocument.h>
#include <Materials/SVM/SvmCompiler.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mx = MaterialX;

PXR_NAMESPACE_OPEN_SCOPE

namespace
{

float FloatValue(const VtValue& value, const float fallback)
{
    if (value.IsHolding<float>())
        return value.UncheckedGet<float>();
    if (value.IsHolding<double>())
        return static_cast<float>(value.UncheckedGet<double>());
    if (value.IsHolding<int>())
        return static_cast<float>(value.UncheckedGet<int>());
    return fallback;
}

glm::vec2 Vec2Value(const VtValue& value, const glm::vec2 fallback)
{
    if (value.IsHolding<GfVec2f>()) {
        const GfVec2f& v = value.UncheckedGet<GfVec2f>();
        return {v[0], v[1]};
    }
    if (value.IsHolding<GfVec2d>()) {
        const GfVec2d& v = value.UncheckedGet<GfVec2d>();
        return {static_cast<float>(v[0]), static_cast<float>(v[1])};
    }
    return fallback;
}

glm::vec3 ColorValue(const VtValue& value, const glm::vec3 fallback)
{
    if (value.IsHolding<GfVec3f>()) {
        const GfVec3f& v = value.UncheckedGet<GfVec3f>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec3d>()) {
        const GfVec3d& v = value.UncheckedGet<GfVec3d>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec4f>()) {
        const GfVec4f& v = value.UncheckedGet<GfVec4f>();
        return {v[0], v[1], v[2]};
    }
    if (value.IsHolding<GfVec4d>()) {
        const GfVec4d& v = value.UncheckedGet<GfVec4d>();
        return {
            static_cast<float>(v[0]), static_cast<float>(v[1]),
            static_cast<float>(v[2])};
    }
    if (value.IsHolding<float>()) {
        const float f = value.UncheckedGet<float>();
        return {f, f, f};
    }
    return fallback;
}

std::string TextureFilePath(const VtValue& value)
{
    if (value.IsHolding<SdfAssetPath>()) {
        const SdfAssetPath& assetPath = value.UncheckedGet<SdfAssetPath>();
        std::string path = assetPath.GetResolvedPath();
        if (path.empty())
            path = assetPath.GetAssetPath();
        return path;
    }
    if (value.IsHolding<std::string>())
        return value.UncheckedGet<std::string>();
    if (value.IsHolding<TfToken>())
        return value.UncheckedGet<TfToken>().GetString();
    return {};
}

// ---------------------------------------------------------------------
// MaterialX ingestion.
//
// Building the MaterialX document graph itself NEVER touches Blender-USD's
// own embedded MaterialX (libusd_ms.so, which hdnoorray also links for pxr
// symbols): only pxr types (HdMaterialNetwork2, SdfPath, VtValue, ...) cross
// that boundary here. MaterialX::Document objects are constructed and
// consumed entirely within NoorRay's own vendored MaterialXCore -- see
// docs/MaterialX.md on why two independently-compiled copies of the same
// namespaced MaterialX types must never trade live objects.
const mx::DocumentPtr& GetSharedMaterialXLibraries()
{
    static const mx::DocumentPtr libraries =
        nr::materialx::loadStandardLibraries(NR_MATERIALX_STDLIB_DIR);
    return libraries;
}

void SetMxInputValue(const mx::NodePtr& node, const std::string& inputName,
    const std::string& mxType, const VtValue& value)
{
    if (mxType == "float")
        node->setInputValue(inputName, FloatValue(value, 0.0f));
    else if (mxType == "integer")
        node->setInputValue(inputName,
            value.IsHolding<int>() ? value.UncheckedGet<int>()
                                    : static_cast<int>(FloatValue(value, 0.0f)));
    else if (mxType == "boolean")
        node->setInputValue(inputName,
            value.IsHolding<bool>() ? value.UncheckedGet<bool>()
                                     : FloatValue(value, 0.0f) != 0.0f);
    else if (mxType == "color3") {
        const glm::vec3 c = ColorValue(value, glm::vec3(0.0f));
        node->setInputValue(inputName, mx::Color3(c.x, c.y, c.z));
    } else if (mxType == "color4") {
        const glm::vec3 c = ColorValue(value, glm::vec3(0.0f));
        node->setInputValue(inputName, mx::Color4(c.x, c.y, c.z, 1.0f));
    } else if (mxType == "vector3" || mxType == "normal" || mxType == "point") {
        const glm::vec3 c = ColorValue(value, glm::vec3(0.0f));
        node->setInputValue(inputName, mx::Vector3(c.x, c.y, c.z));
    } else if (mxType == "vector2") {
        const glm::vec2 c = Vec2Value(value, glm::vec2(0.0f));
        node->setInputValue(inputName, mx::Vector2(c.x, c.y));
    } else if (mxType == "vector4") {
        const glm::vec3 c = ColorValue(value, glm::vec3(0.0f));
        node->setInputValue(inputName, mx::Vector4(c.x, c.y, c.z, 0.0f));
    } else if (mxType == "filename") {
        const std::string path = TextureFilePath(value);
        if (!path.empty())
            node->setInputValue(inputName, path, mxType);
    } else if (mxType == "string") {
        std::string s;
        if (value.IsHolding<std::string>())
            s = value.UncheckedGet<std::string>();
        else if (value.IsHolding<TfToken>())
            s = value.UncheckedGet<TfToken>().GetString();
        if (!s.empty())
            node->setInputValue(inputName, s, mxType);
    }
    // Any other declared type (matrices, arrays, ...) is outside what
    // Blender's MaterialX export or NoorRay's supported surface profile
    // uses today; the nodedef's own default value stands.
}

// Recursively translates one HdMaterialNetwork2 node (and everything it
// depends on) into NoorRay's own mx::Document, resolving each node's
// MaterialX category/type from its nodedef in `libraries` -- the same
// technique OpenUSD's own hdMtlx uses (HdMtlxCreateMtlxDocumentFromHdNetwork),
// reimplemented against NoorRay's vendored MaterialX instead of calling into
// Blender-USD's embedded copy (see this file's header comment on why).
mx::NodePtr TranslateNode(const HdMaterialNetwork2& network, const SdfPath& path,
    const mx::DocumentPtr& doc, const mx::DocumentPtr& libraries,
    std::map<SdfPath, mx::NodePtr>& created, std::set<SdfPath>& visiting)
{
    const auto existing = created.find(path);
    if (existing != created.end())
        return existing->second;
    if (visiting.count(path))
        return nullptr; // cyclic graph -- not a valid material, stop here.

    const auto nodeIt = network.nodes.find(path);
    if (nodeIt == network.nodes.end())
        return nullptr;
    const HdMaterialNode2& hdNode = nodeIt->second;

    const mx::NodeDefPtr nodeDef = libraries->getNodeDef(hdNode.nodeTypeId.GetString());
    if (!nodeDef) {
        TF_WARN("hdNoorRay: no MaterialX nodedef for node type '%s' at %s -- skipping",
            hdNode.nodeTypeId.GetText(), path.GetText());
        return nullptr;
    }

    visiting.insert(path);

    // Use deterministic ordinal names so equivalent graphs share cache keys
    // without embedding USD instance identity in the document.
    const std::string nodeName =
        "nr_node_" + std::to_string(doc->getNodes().size());
    const mx::NodePtr mxNode =
        doc->addNode(nodeDef->getNodeString(), nodeName, nodeDef->getType());

    for (const mx::InputPtr& declInput : nodeDef->getActiveInputs()) {
        const std::string& inputName = declInput->getName();
        const TfToken inputToken(inputName);

        const auto connIt = hdNode.inputConnections.find(inputToken);
        if (connIt != hdNode.inputConnections.end() && !connIt->second.empty()) {
            const HdMaterialConnection2& connection = connIt->second.front();
            const mx::NodePtr upstream = TranslateNode(
                network, connection.upstreamNode, doc, libraries, created, visiting);
            if (upstream) {
                mxNode->setConnectedNode(inputName, upstream);
                // Hydra always names a connection's source output (typically
                // "out" even for an ordinary single-output node), but
                // MaterialX only accepts an explicit output string on a
                // connection to a node whose nodedef actually declares more
                // than one output -- setting one on a single-output node
                // fails validation ("Multi-output type expected"). Look the
                // upstream node's own declared output count up rather than
                // trusting the connection's output name is meaningful here.
                const auto upstreamNodeIt = network.nodes.find(connection.upstreamNode);
                const mx::NodeDefPtr upstreamNodeDef = upstreamNodeIt != network.nodes.end()
                    ? libraries->getNodeDef(upstreamNodeIt->second.nodeTypeId.GetString())
                    : nullptr;
                if (!connection.upstreamOutputName.IsEmpty() && upstreamNodeDef
                    && upstreamNodeDef->getActiveOutputs().size() > 1) {
                    if (const mx::InputPtr mxInput = mxNode->getInput(inputName))
                        mxInput->setOutputString(connection.upstreamOutputName.GetString());
                }
            }
            continue;
        }

        const auto paramIt = hdNode.parameters.find(inputToken);
        if (paramIt != hdNode.parameters.end())
            SetMxInputValue(mxNode, inputName, declInput->getType(), paramIt->second);
    }

    visiting.erase(path);
    created[path] = mxNode;
    return mxNode;
}

// Pre-load textures and hand a fully authored document to the shared
// asynchronous compiler tail. Keeping this path common to direct XML and
// Hydra graph translation makes both ingestion modes share all caches.
bool QueueMaterialXDocument(mx::DocumentPtr doc, const SdfPath& materialId,
    HdNoorRayRenderParam& param)
{
    const mx::DocumentPtr& libraries = GetSharedMaterialXLibraries();

    // Pre-load every <image> node's texture (with the color/data encoding
    // collectImageNodes knows -- see MaterialXImageNode's comment) before
    // attaching the standard library. traverseTree() consequently only walks
    // this material's own handful of elements.
    auto resolvedTextures = std::make_shared<
        std::unordered_map<std::string, std::uint32_t>>();
    std::vector<TextureRef> parsedTextures;
    for (const nr::materialx::MaterialXImageNode& image :
        nr::materialx::collectImageNodes(doc)) {
        TextureRef texture = param.GetOrCreateTexture(
            image.rawFilePath,
            image.colorSpace == nr::materialx::MaterialXImageColorSpace::Srgb
                ? TextureEncoding::Srgb8
                : TextureEncoding::Linear8);
        if (!texture.isValid())
            continue;
        (*resolvedTextures)[image.rawFilePath] =
            static_cast<std::uint32_t>(texture.index());
        parsedTextures.push_back(std::move(texture));
    }

    // A data library is resolved transparently by MaterialX but remains one
    // immutable process-wide document. importLibrary() deep-copies the whole
    // stdlib into every user document and is prohibitively expensive for
    // scenes containing thousands of materials.
    doc->setDataLibrary(libraries);
    // The shared_ptr is copied for the async compile before the document is
    // handed to the compile queue (which publishes it to the Scene slot).
    const mx::DocumentPtr compileDocument = doc;
    param.QueueMaterialCompilation(materialId, std::move(doc),
        std::move(parsedTextures),
        [compileDocument, resolvedTextures]() {
            HdNoorRayRenderParam::MaterialCompilationOutput output;
            nr::svm::SvmCompiler compiler;
            output.program = compiler.compile(compileDocument, {}, *resolvedTextures);
            return output;
        });
    return true;
}

// Compiles one MaterialX surface terminal (standard_surface, open_pbr_surface,
// ...) reached from the full HdMaterialNetwork2 graph.
bool QueueMaterialXNetwork(const HdMaterialNetwork2& network,
    const SdfPath& materialId, const SdfPath& terminalPath,
    HdNoorRayRenderParam& param)
{
    const mx::DocumentPtr& libraries = GetSharedMaterialXLibraries();

    mx::DocumentPtr doc = mx::createDocument();
    std::map<SdfPath, mx::NodePtr> created;
    std::set<SdfPath> visiting;
    const mx::NodePtr terminalNode =
        TranslateNode(network, terminalPath, doc, libraries, created, visiting);
    if (!terminalNode) {
        TF_WARN("hdNoorRay: could not translate MaterialX terminal at %s",
            terminalPath.GetText());
        return false;
    }
    const mx::NodePtr materialNode =
        doc->addNode("surfacematerial", "NR_hdnoorray_material", "material");
    materialNode->setConnectedNode("surfaceshader", terminalNode);
    return QueueMaterialXDocument(
        std::move(doc), materialId, param);
}

bool QueueMaterialXXml(const std::string& xml, const SdfPath& materialId,
    HdNoorRayRenderParam& param)
{
    mx::DocumentPtr doc = mx::createDocument();
    mx::readFromXmlString(doc, xml);
    return QueueMaterialXDocument(
        std::move(doc), materialId, param);
}

} // namespace

const mx::DocumentPtr& GetSharedNativeFallbackMaterial()
{
    static const mx::DocumentPtr fallback = []() {
        MaterialAuthoring authoring;
        authoring.albedo = glm::vec3(0.8f);
        authoring.roughness = 0.5f;
        return nr::materialx::documentFromAuthoring(authoring);
    }();
    return fallback;
}

HdNoorRayMaterial::HdNoorRayMaterial(const SdfPath& id)
    : HdMaterial(id)
{
}

void HdNoorRayMaterial::Sync(
    HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits)
{
    static const bool profileSync = std::getenv("NR_PROFILE_SYNC") != nullptr;
    const auto syncStart = profileSync
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const auto finishSync = [&]() {
        *dirtyBits = Clean;
        if (profileSync) {
            const double milliseconds =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - syncStart).count();
            fprintf(stderr, "NR_PROFILE material %.3f %s\n",
                milliseconds, GetId().GetText());
        }
    };

    const mx::DocumentPtr& fallbackMaterial = GetSharedNativeFallbackMaterial();
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    struct MaterialSyncSlot
    {
        explicit MaterialSyncSlot(HdNoorRayRenderParam& value)
            : param(value)
        {
            param.AcquireMaterialSyncSlot();
        }
        ~MaterialSyncSlot() { param.ReleaseMaterialSyncSlot(); }
        HdNoorRayRenderParam& param;
    } materialSyncSlot(param);
    bool compilationQueued = false;

    const std::optional<HdNoorRayRenderParam::MaterialXDocument> customDocument =
        param.GetMaterialXDocument(GetId().GetName());
    if (customDocument && !customDocument->IsTombstone()) {
        if (usingCustomDocument_
            && customDocumentRevision_ == customDocument->revision
            && customDocumentHash_ == customDocument->hash) {
            finishSync();
            return;
        }

        try {
            compilationQueued = QueueMaterialXXml(
                *customDocument->xml, GetId(), param);
        } catch (const std::exception& error) {
            TF_WARN("hdNoorRay: could not parse custom MaterialX for %s: %s",
                GetId().GetText(), error.what());
        }
        if (compilationQueued) {
            usingCustomDocument_ = true;
            customDocumentHash_ = customDocument->hash;
            customDocumentRevision_ = customDocument->revision;
            finishSync();
            return;
        }
    }

    // No direct document (or an explicit empty tombstone): retain the native
    // Hydra resource path as a compatibility fallback.
    usingCustomDocument_ = false;
    const VtValue resource = sceneDelegate->GetMaterialResource(GetId());
    if (getenv("NR_MATERIALX_DEBUG")) {
        fprintf(stderr, "hdNoorRay: Sync(%s) resource type='%s' isEmpty=%d\n",
            GetId().GetText(), resource.GetTypeName().c_str(), resource.IsEmpty());
    }

    // Blender's Hydra render engine hands materials over as the older,
    // relationship-based HdMaterialNetworkMap, not HdMaterialNetwork2 --
    // confirmed empirically (NR_MATERIALX_DEBUG=1 logs the resource's held
    // type). Converting up front, through USD's own (MaterialX-free, so
    // there is no ABI concern -- see this file's header comment) conversion
    // utility means the real graph translator below always runs, on either
    // representation, instead of only ever seeing the flattened fallback.
    HdMaterialNetwork2 converted;
    const HdMaterialNetwork2* network = nullptr;
    if (resource.IsHolding<HdMaterialNetwork2>()) {
        network = &resource.UncheckedGet<HdMaterialNetwork2>();
    } else if (resource.IsHolding<HdMaterialNetworkMap>()) {
        const HdMaterialNetworkMap& networkMap = resource.UncheckedGet<HdMaterialNetworkMap>();
        if (getenv("NR_MATERIALX_DEBUG")) {
            for (const auto& [context, rawNetwork] : networkMap.map) {
                fprintf(stderr, "hdNoorRay: raw map context='%s' nodeCount=%zu\n",
                    context.GetText(), rawNetwork.nodes.size());
                for (const HdMaterialNode& node : rawNetwork.nodes)
                    fprintf(stderr, "hdNoorRay:   raw node %s : identifier='%s'\n",
                        node.path.GetText(), node.identifier.GetText());
            }
            std::string terminalsStr;
            for (const SdfPath& terminalPath : networkMap.terminals)
                terminalsStr += terminalPath.GetString() + " ";
            fprintf(stderr, "hdNoorRay: raw terminals: [ %s]\n", terminalsStr.c_str());
        }
        converted = HdConvertToHdMaterialNetwork2(networkMap);
        network = &converted;
    }

    if (network != nullptr) {
        const bool debugMaterialX = getenv("NR_MATERIALX_DEBUG") != nullptr;
        if (debugMaterialX) {
            std::string terminalsStr;
            for (const auto& [token, connection] : network->terminals)
                terminalsStr += token.GetString() + "->" + connection.upstreamNode.GetString() + " ";
            fprintf(stderr, "hdNoorRay: material %s terminals: [ %s]\n", GetId().GetText(), terminalsStr.c_str());
            for (const auto& [path, node] : network->nodes)
                fprintf(stderr, "hdNoorRay:   node %s : nodeTypeId='%s'\n",
                    path.GetText(), node.nodeTypeId.GetText());
        }

        const auto surfaceTerminal = network->terminals.find(TfToken("surface"));
        const bool hasMaterialXTerminal = surfaceTerminal != network->terminals.end()
            && network->nodes.count(surfaceTerminal->second.upstreamNode) != 0
            && GetSharedMaterialXLibraries()->getNodeDef(
                   network->nodes.at(surfaceTerminal->second.upstreamNode).nodeTypeId.GetString())
                != nullptr;

        if (debugMaterialX && surfaceTerminal != network->terminals.end()
            && network->nodes.count(surfaceTerminal->second.upstreamNode) != 0) {
            const std::string& typeId =
                network->nodes.at(surfaceTerminal->second.upstreamNode).nodeTypeId.GetString();
            fprintf(stderr, "hdNoorRay: terminal nodeTypeId='%s' nodedef-found=%d\n",
                typeId.c_str(),
                GetSharedMaterialXLibraries()->getNodeDef(typeId) != nullptr);
        }

        if (hasMaterialXTerminal) {
            compilationQueued = QueueMaterialXNetwork(*network, GetId(),
                surfaceTerminal->second.upstreamNode, param);
        }
        // No translatable MaterialX terminal: fall back to the native grey
        // material (see above), rather than parsing whatever flattened
        // UsdPreviewSurface network Hydra also handed over.
    }
    if (!compilationQueued) {
        param.PublishFallbackMaterial(GetId(), fallbackMaterial);
    }
    finishSync();
}

void HdNoorRayMaterial::Finalize(HdRenderParam* renderParam)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    param.ReleaseMaterial(GetId());
}

HdDirtyBits HdNoorRayMaterial::GetInitialDirtyBitsMask() const
{
    return AllDirty;
}

PXR_NAMESPACE_CLOSE_SCOPE
