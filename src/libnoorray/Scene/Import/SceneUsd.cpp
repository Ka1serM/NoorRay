#include "Scene/Import/SceneUsd.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include <MaterialXCore/Document.h>
#include <MaterialXCore/Node.h>
#include <MaterialXFormat/XmlIo.h>

#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/shapingAPI.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>

#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/FisheyeCamera.h"
#include "Rendering/Camera/HybridPsfCamera.h"
#include "Rendering/Camera/OrthographicCamera.h"
#include "Rendering/Camera/PerspectiveCamera.h"
#include "Rendering/Camera/RealisticCamera.h"
#include "Rendering/Camera/ThinLensCamera.h"
#include "Rendering/Lighting/DirectionalLight.h"
#include "Rendering/Lighting/PointLight.h"
#include "Rendering/Lighting/RectLight.h"
#include "Rendering/Lighting/SpotLight.h"
#include "Materials/MaterialX/MaterialXDocument.h"
#include "Geometry/Mesh/Assets/MeshAsset.h"
#include "Geometry/Mesh/VertexColor.h"
#include "Geometry/Mesh/Transform.h"
#include "Scene/Objects/LightInstance.h"
#include "Scene/Objects/MeshInstance.h"
#include "Scene/Scene.h"
#include "Scene/Import/SceneFile.h"
#include "Scene/Import/SceneImporter.h"
#include "Scene/SceneObject.h"
#include "Scene/Objects/GaussianInstance.h"

namespace pxr = PXR_NS;
namespace mx = MaterialX;

namespace {

using pxr::GfVec3f;
using pxr::GfVec3d;
using pxr::SdfPath;
using pxr::TfToken;
using pxr::UsdAttribute;
using pxr::UsdPrim;
using pxr::UsdStageRefPtr;
using pxr::UsdTimeCode;
using pxr::VtArray;
using pxr::VtIntArray;
using pxr::VtStringArray;

constexpr const char* kSceneRoot = "/NoorRayScene";
constexpr const char* kMaterialsRoot = "/NoorRayScene/Materials";
constexpr const char* kMaterialXml = "nr:materialX";

std::string lower(std::string value)
{
    std::ranges::transform(value, value.begin(), [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string safeName(std::string value, const std::string& fallback)
{
    if (value.empty()) value = fallback;
    for (char& c : value)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
    if (value.empty() || std::isdigit(static_cast<unsigned char>(value.front())))
        value = "_" + value;
    return value;
}

template <typename T>
void setAttr(const UsdPrim& prim, const char* name,
    const pxr::SdfValueTypeName& type, const T& value)
{
    prim.CreateAttribute(TfToken(name), type).Set(value);
}

template <typename T>
bool getAttr(const UsdPrim& prim, const char* name, T* value)
{
    const UsdAttribute attr = prim.GetAttribute(TfToken(name));
    return attr && attr.Get(value, UsdTimeCode::Default());
}

glm::vec3 toGlm(const GfVec3f& value) { return {value[0], value[1], value[2]}; }
GfVec3f toUsd(const glm::vec3& value) { return {value.x, value.y, value.z}; }

void writeTransform(const pxr::UsdGeomXformable& object, const SceneObject& source)
{
    object.AddTranslateOp().Set(GfVec3d(source.getPosition().x,
        source.getPosition().y, source.getPosition().z));
    object.AddRotateXYZOp().Set(toUsd(source.getRotationEuler()));
    object.AddScaleOp().Set(toUsd(source.getScale()));
}

Transform readTransform(const UsdPrim& prim)
{
    pxr::UsdGeomXformable xform(prim);
    GfVec3d translation{};
    GfVec3f rotation{};
    GfVec3f scale{1.0f};
    bool resetsXformStack = false;
    for (const pxr::UsdGeomXformOp& op : xform.GetOrderedXformOps(&resetsXformStack)) {
        if (op.GetOpType() == pxr::UsdGeomXformOp::TypeTranslate)
            op.Get(&translation);
        else if (op.GetOpType() == pxr::UsdGeomXformOp::TypeRotateXYZ)
            op.Get(&rotation);
        else if (op.GetOpType() == pxr::UsdGeomXformOp::TypeScale)
            op.Get(&scale);
    }
    return Transform({static_cast<float>(translation[0]), static_cast<float>(translation[1]),
        static_cast<float>(translation[2])}, toGlm(rotation), toGlm(scale));
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Failed to open MaterialX file: " + path.string());
    std::ostringstream result;
    result << input.rdbuf();
    return result.str();
}

mx::DocumentPtr materialDocument(const Scene& scene, const uint32_t slot,
    const std::filesystem::path& usdPath)
{
    const auto& documents = scene.getMaterialXDocuments();
    const auto& paths = scene.getMaterialXSourcePaths();
    if (slot < documents.size() && documents[slot])
        return documents[slot];

    if (slot < paths.size() && !paths[slot].empty()) {
        std::filesystem::path source(paths[slot]);
        if (source.is_relative()) source = usdPath.parent_path() / source;
        const auto document = mx::createDocument();
        mx::readFromXmlString(document, readFile(source));
        return document;
    }
    return nr::materialx::defaultMaterial();
}

std::string materialXml(const mx::DocumentPtr& document)
{
    mx::XmlWriteOptions options;
    options.writeXIncludeEnable = false;
    return mx::writeToXmlString(document, &options);
}

pxr::SdfValueTypeName usdType(const std::string& type)
{
    if (type == "float") return pxr::SdfValueTypeNames->Float;
    if (type == "integer") return pxr::SdfValueTypeNames->Int;
    if (type == "boolean") return pxr::SdfValueTypeNames->Bool;
    if (type == "color3") return pxr::SdfValueTypeNames->Color3f;
    if (type == "color4") return pxr::SdfValueTypeNames->Color4f;
    if (type == "vector2") return pxr::SdfValueTypeNames->Float2;
    if (type == "vector3" || type == "normal" || type == "point")
        return pxr::SdfValueTypeNames->Float3;
    if (type == "filename") return pxr::SdfValueTypeNames->Asset;
    return pxr::SdfValueTypeNames->String;
}

std::vector<float> numbers(std::string value)
{
    for (char& c : value) if (c == ',') c = ' ';
    std::stringstream stream(value);
    std::vector<float> values;
    for (float number; stream >> number;) values.push_back(number);
    return values;
}

void setUsdInput(pxr::UsdShadeShader& shader, const mx::InputPtr& input)
{
    const std::string type = input->getType();
    const auto inputValue = shader.CreateInput(TfToken(input->getName()), usdType(type));
    if (!inputValue) return;
    if (const mx::NodePtr upstream = input->getConnectedNode()) {
        const std::string upstreamName = safeName(upstream->getName(), "node");
        const std::string outputName = input->hasOutputString() && !input->getOutputString().empty()
            ? input->getOutputString() : "out";
        pxr::UsdShadeShader upstreamShader(
            shader.GetPrim().GetStage()->GetPrimAtPath(
                shader.GetPrim().GetPath().GetParentPath().AppendChild(TfToken(upstreamName))));
        if (upstreamShader) {
            pxr::UsdShadeOutput output = upstreamShader.GetOutput(TfToken(outputName));
            if (!output) output = upstreamShader.GetOutput(TfToken("surface"));
            if (output) inputValue.ConnectToSource(output);
        }
        return;
    }
    if (!input->getValue()) return;
    const std::vector<float> values = numbers(input->getValueString());
    if (type == "float" && !values.empty()) inputValue.Set(values[0]);
    else if (type == "integer" && !values.empty()) inputValue.Set(static_cast<int>(values[0]));
    else if (type == "boolean") inputValue.Set(input->getValueString() == "true");
    else if (type == "color3" && values.size() >= 3)
        inputValue.Set(pxr::GfVec3f(values[0], values[1], values[2]));
    else if ((type == "vector2") && values.size() >= 2)
        inputValue.Set(pxr::GfVec2f(values[0], values[1]));
    else if ((type == "color4") && values.size() >= 4)
        inputValue.Set(pxr::GfVec4f(values[0], values[1], values[2], values[3]));
    else if ((type == "vector3" || type == "normal" || type == "point") && values.size() >= 3)
        inputValue.Set(pxr::GfVec3f(values[0], values[1], values[2]));
    else if (type == "filename")
        inputValue.Set(pxr::SdfAssetPath(input->getValueString()));
    else
        inputValue.Set(input->getValueString());
}

void writeMaterialNetwork(const mx::DocumentPtr& document,
    pxr::UsdShadeMaterial material)
{
    const pxr::UsdStageRefPtr stage = material.GetPrim().GetStage();
    std::unordered_map<std::string, pxr::UsdShadeShader> shaders;
    for (const mx::NodePtr& node : document->getNodes()) {
        const std::string nodeName = safeName(node->getName(), "node");
        const pxr::SdfPath path = material.GetPath().AppendChild(TfToken(nodeName));
        pxr::UsdShadeShader shader = pxr::UsdShadeShader::Define(stage, path);
        shader.CreateIdAttr().Set(TfToken("ND_" + node->getCategory() + "_" + node->getType()));
        shader.CreateOutput(TfToken(node->getType() == "surfaceshader" ? "surface" : "out"),
            node->getType() == "surfaceshader" ? pxr::SdfValueTypeNames->Token
                                                 : usdType(node->getType()));
        shaders.emplace(node->getName(), shader);
    }
    for (const mx::NodePtr& node : document->getNodes()) {
        const auto found = shaders.find(node->getName());
        if (found == shaders.end()) continue;
        for (const mx::InputPtr& input : node->getInputs()) setUsdInput(found->second, input);
        if (node->getType() == "surfaceshader")
            material.CreateSurfaceOutput(TfToken("mtlx")).ConnectToSource(
                found->second.GetOutput(TfToken("surface")));
    }
}

struct MaterialTable {
    std::unordered_map<uint32_t, pxr::UsdShadeMaterial> bySlot;
    std::unordered_map<std::string, MaterialRef> loaded;
};

pxr::UsdShadeMaterial writeMaterial(const Scene& scene, const uint32_t slot,
    const UsdStageRefPtr& stage, const std::filesystem::path& usdPath,
    MaterialTable& table)
{
    if (const auto found = table.bySlot.find(slot); found != table.bySlot.end())
        return found->second;
    const SdfPath path = SdfPath(kMaterialsRoot).AppendChild(
        TfToken("Material_" + std::to_string(slot)));
    const auto material = pxr::UsdShadeMaterial::Define(stage, path);
    const mx::DocumentPtr document = materialDocument(scene, slot, usdPath);
    setAttr(material.GetPrim(), kMaterialXml, pxr::SdfValueTypeNames->String,
        materialXml(document));
    writeMaterialNetwork(document, material);
    table.bySlot.emplace(slot, material);
    return material;
}

void writeMesh(const Scene& scene, const MeshInstance& instance,
    const UsdStageRefPtr& stage, const SdfPath& path,
    const std::filesystem::path& usdPath, MaterialTable& materials)
{
    const MeshAsset& asset = instance.getMeshAsset();
    const pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh::Define(stage, path);
    writeTransform(mesh, instance);
    mesh.CreateSubdivisionSchemeAttr().Set(TfToken("none"));

    VtArray<GfVec3f> points;
    VtArray<GfVec3f> normals;
    points.reserve(asset.getVertices().size());
    normals.reserve(asset.getVertices().size());
    for (const Vertex& vertex : asset.getVertices()) {
        points.push_back(toUsd(vertex.position));
        normals.push_back(toUsd(vertex.normal));
    }
    VtIntArray indices;
    VtIntArray counts;
    for (size_t i = 0; i + 2 < asset.getIndices().size(); i += 3) {
        counts.push_back(3);
        indices.push_back(static_cast<int>(asset.getIndices()[i]));
        indices.push_back(static_cast<int>(asset.getIndices()[i + 1]));
        indices.push_back(static_cast<int>(asset.getIndices()[i + 2]));
    }
    mesh.CreatePointsAttr().Set(points);
    mesh.CreateFaceVertexCountsAttr().Set(counts);
    mesh.CreateFaceVertexIndicesAttr().Set(indices);
    mesh.CreateNormalsAttr().Set(normals);
    VtArray<pxr::GfVec2f> uvs;
    VtArray<GfVec3f> tangents;
    VtArray<float> tangentSigns;
    VtArray<GfVec3f> displayColors;
    VtArray<float> displayOpacities;
    uvs.reserve(asset.getVertices().size());
    tangents.reserve(asset.getVertices().size());
    tangentSigns.reserve(asset.getVertices().size());
    displayColors.reserve(asset.getVertices().size());
    displayOpacities.reserve(asset.getVertices().size());
    for (const Vertex& vertex : asset.getVertices()) {
        uvs.push_back({vertex.uv.x, vertex.uv.y});
        tangents.push_back(toUsd(vertex.tangent));
        tangentSigns.push_back(vertex.tangentSign);
        const glm::vec4 color = nr::vertex_color::unpackLinear(vertex.color);
        displayColors.push_back({color.r, color.g, color.b});
        displayOpacities.push_back(color.a);
    }
    pxr::UsdGeomPrimvarsAPI(mesh).CreateNonIndexedPrimvar(
        TfToken("st"), pxr::SdfValueTypeNames->TexCoord2fArray, uvs,
        pxr::UsdGeomTokens->vertex);
    pxr::UsdGeomPrimvarsAPI(mesh).CreateNonIndexedPrimvar(
        pxr::UsdGeomTokens->tangents, pxr::SdfValueTypeNames->Vector3fArray,
        tangents, pxr::UsdGeomTokens->vertex);
    mesh.CreateDisplayColorPrimvar(pxr::UsdGeomTokens->vertex).Set(displayColors);
    mesh.CreateDisplayOpacityPrimvar(pxr::UsdGeomTokens->vertex).Set(displayOpacities);
    setAttr(mesh.GetPrim(), "nr:tangentSigns", pxr::SdfValueTypeNames->FloatArray,
        tangentSigns);
    setAttr(mesh.GetPrim(), "nr:materialIndices", pxr::SdfValueTypeNames->IntArray,
        [&] { VtIntArray result; for (uint32_t id : asset.getMaterialIds()) result.push_back(static_cast<int>(id)); return result; }());
    setAttr(mesh.GetPrim(), "nr:faceMaterialIndices", pxr::SdfValueTypeNames->IntArray,
        [&] { VtIntArray result; for (const Face& face : asset.getFaces()) result.push_back(face.materialIndex); return result; }());

    VtStringArray materialPaths;
    for (size_t slot = 0; slot < asset.getMaterialCount(); ++slot) {
        const uint32_t materialSlot = asset.getMaterialIds()[slot];
        const auto material = writeMaterial(scene, materialSlot, stage, usdPath, materials);
        materialPaths.push_back(material.GetPath().GetString());
    }
    setAttr(mesh.GetPrim(), "nr:materialPaths", pxr::SdfValueTypeNames->StringArray, materialPaths);
    if (asset.getMaterialCount() == 1)
        pxr::UsdShadeMaterialBindingAPI(mesh.GetPrim()).Bind(
            writeMaterial(scene, asset.getMaterialIds()[0], stage, usdPath, materials));
}

void writeCamera(const CameraInstance& instance, const UsdStageRefPtr& stage,
    const SdfPath& path, const bool active)
{
    const Camera* camera = instance.getCamera();
    const pxr::UsdGeomCamera usdCamera = pxr::UsdGeomCamera::Define(stage, path);
    writeTransform(usdCamera, instance);
    usdCamera.CreateProjectionAttr().Set(TfToken(
        instance.getProjectionType() == CameraProjectionType::Orthographic ? "orthographic" : "perspective"));
    usdCamera.CreateFocalLengthAttr().Set(camera->getFocalLengthMm());
    usdCamera.CreateFocusDistanceAttr().Set(camera->getFocusDistanceCm() * 10.0f);
    usdCamera.CreateHorizontalApertureAttr().Set(camera->getSensor().width());
    usdCamera.CreateVerticalApertureAttr().Set(camera->getSensor().height());
    setAttr(usdCamera.GetPrim(), "nr:active", pxr::SdfValueTypeNames->Bool, active);
    setAttr(usdCamera.GetPrim(), "nr:exposure", pxr::SdfValueTypeNames->Float, camera->exposure);
    const auto resolution = camera->getSensor().resolution();
    setAttr(usdCamera.GetPrim(), "nr:resolution", pxr::SdfValueTypeNames->Int2,
        pxr::GfVec2i(resolution.x, resolution.y));
    setAttr(usdCamera.GetPrim(), "nr:sensorType", pxr::SdfValueTypeNames->String,
        camera->getSensor().getType() == SensorType::Rectangular ? "rectangular" :
        camera->getSensor().getType() == SensorType::ScatterPsf ? "scatter_psf" : "gather_psf");
}

void writeLight(const LightInstance& instance, const UsdStageRefPtr& stage, const SdfPath& path)
{
    UsdPrim prim;
    switch (instance.lightType) {
    case LightInstance::TypePoint:
    case LightInstance::TypeSpot:
        prim = pxr::UsdLuxSphereLight::Define(stage, path).GetPrim();
        break;
    case LightInstance::TypeRect:
        prim = pxr::UsdLuxRectLight::Define(stage, path).GetPrim();
        break;
    case LightInstance::TypeDirectional:
        prim = pxr::UsdLuxDistantLight::Define(stage, path).GetPrim();
        break;
    default:
        prim = stage->DefinePrim(path);
        break;
    }
    writeTransform(pxr::UsdGeomXformable(prim), instance);
    const pxr::UsdLuxLightAPI usdLight(prim);
    usdLight.CreateColorAttr().Set(toUsd(instance.getColor()));
    setAttr(prim, "nr:lightType", pxr::SdfValueTypeNames->Int, instance.lightType);
    setAttr(prim, "nr:color", pxr::SdfValueTypeNames->Color3f, toUsd(instance.getColor()));
    const float intensity = std::visit([](const auto& light) { return light.intensity; }, instance.getLightData());
    usdLight.CreateIntensityAttr().Set(intensity);
    setAttr(prim, "nr:intensity", pxr::SdfValueTypeNames->Float, intensity);

    if (const auto* point = std::get_if<PointLight>(&instance.getLightData()))
        pxr::UsdLuxSphereLight(prim).CreateRadiusAttr().Set(point->softRadius);
    else if (const auto* spot = std::get_if<SpotLight>(&instance.getLightData())) {
        pxr::UsdLuxSphereLight(prim).CreateRadiusAttr().Set(spot->softRadius);
        const auto shaping = pxr::UsdLuxShapingAPI::Apply(prim);
        if (shaping)
            shaping.CreateShapingConeAngleAttr().Set(spot->outerConeAngle);
    } else if (const auto* rect = std::get_if<RectLight>(&instance.getLightData())) {
        const pxr::UsdLuxRectLight usdRect(prim);
        usdRect.CreateWidthAttr().Set(rect->width);
        usdRect.CreateHeightAttr().Set(rect->height);
    } else if (const auto* directional = std::get_if<DirectionalLight>(&instance.getLightData()))
        pxr::UsdLuxDistantLight(prim).CreateAngleAttr().Set(directional->softAngle);
}

void writeObject(const Scene& scene, const std::shared_ptr<SceneObject>& object,
    const UsdStageRefPtr& stage, const SdfPath& parent, const std::filesystem::path& usdPath,
    MaterialTable& materials, const CameraInstance* activeCamera)
{
    if (!object) return;
    const SdfPath path = parent.AppendChild(TfToken(safeName(object->getName(), "Object")));
    if (const auto mesh = std::dynamic_pointer_cast<MeshInstance>(object))
        writeMesh(scene, *mesh, stage, path, usdPath, materials);
    else if (const auto camera = std::dynamic_pointer_cast<CameraInstance>(object))
        writeCamera(*camera, stage, path, camera.get() == activeCamera);
    else if (const auto light = std::dynamic_pointer_cast<LightInstance>(object))
        writeLight(*light, stage, path);
    else if (const auto gaussian = std::dynamic_pointer_cast<GaussianInstance>(object)) {
        const UsdPrim prim = stage->DefinePrim(path);
        writeTransform(pxr::UsdGeomXformable(prim), *gaussian);
        setAttr(prim, "nr:sourceType", pxr::SdfValueTypeNames->String, "gaussian");
        setAttr(prim, "nr:sourcePath", pxr::SdfValueTypeNames->String, gaussian->getGaussianAsset().getPath());
    } else {
        const UsdPrim prim = stage->DefinePrim(path);
        writeTransform(pxr::UsdGeomXformable(prim), *object);
    }
    const UsdPrim prim = stage->GetPrimAtPath(path);
    setAttr(prim, "nr:sourceType", pxr::SdfValueTypeNames->String, object->getSourceType());
    setAttr(prim, "nr:sourcePath", pxr::SdfValueTypeNames->String, object->getSourcePath());
}

mx::DocumentPtr readMaterial(const UsdPrim& prim)
{
    std::string xml;
    if (!getAttr(prim, kMaterialXml, &xml) || xml.empty())
        return nr::materialx::defaultMaterial();
    const mx::DocumentPtr document = mx::createDocument();
    mx::readFromXmlString(document, xml);
    return document;
}

MaterialRef loadMaterial(const UsdStageRefPtr& stage, const std::string& path,
    Scene& scene, MaterialTable& table)
{
    if (const auto found = table.loaded.find(path); found != table.loaded.end())
        return found->second;
    const MaterialRef material = scene.addMaterial(readMaterial(stage->GetPrimAtPath(SdfPath(path))));
    table.loaded.emplace(path, material);
    return material;
}

void readMesh(Scene& scene, const pxr::UsdGeomMesh& mesh, const std::string& name,
    const Transform& transform, const UsdStageRefPtr& stage, MaterialTable& materials)
{
    VtArray<GfVec3f> points, normals;
    VtIntArray indices, counts;
    mesh.GetPointsAttr().Get(&points);
    mesh.GetNormalsAttr().Get(&normals);
    mesh.GetFaceVertexIndicesAttr().Get(&indices);
    mesh.GetFaceVertexCountsAttr().Get(&counts);
    VtArray<pxr::GfVec2f> uvs;
    VtArray<GfVec3f> tangents;
    VtArray<float> tangentSigns;
    VtArray<GfVec3f> displayColors;
    VtArray<float> displayOpacities;
    const pxr::UsdGeomPrimvar stPrimvar =
        pxr::UsdGeomPrimvarsAPI(mesh).GetPrimvar(TfToken("st"));
    const pxr::UsdGeomPrimvar tangentPrimvar =
        pxr::UsdGeomPrimvarsAPI(mesh).GetPrimvar(pxr::UsdGeomTokens->tangents);
    const pxr::UsdGeomPrimvar colorPrimvar = mesh.GetDisplayColorPrimvar();
    const pxr::UsdGeomPrimvar opacityPrimvar = mesh.GetDisplayOpacityPrimvar();
    getAttr(mesh.GetPrim(), "nr:tangentSigns", &tangentSigns);
    const bool hasVertexUvs = stPrimvar
        && stPrimvar.GetInterpolation() == pxr::UsdGeomTokens->vertex
        && stPrimvar.Get(&uvs)
        && uvs.size() == points.size();
    const bool hasVertexTangents = tangentPrimvar
        && tangentPrimvar.GetInterpolation() == pxr::UsdGeomTokens->vertex
        && tangentPrimvar.Get(&tangents)
        && tangents.size() == points.size();
    const bool hasVertexColors = colorPrimvar
        && colorPrimvar.GetInterpolation() == pxr::UsdGeomTokens->vertex
        && colorPrimvar.Get(&displayColors)
        && displayColors.size() == points.size();
    const bool hasVertexOpacities = opacityPrimvar
        && opacityPrimvar.GetInterpolation() == pxr::UsdGeomTokens->vertex
        && opacityPrimvar.Get(&displayOpacities)
        && displayOpacities.size() == points.size();
    const bool hasVertexTangentSigns = tangentSigns.size() == points.size();
    std::vector<Vertex> vertices(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        vertices[i].position = toGlm(points[i]);
        vertices[i].normal = i < normals.size() ? toGlm(normals[i]) : glm::vec3(0.0f, 1.0f, 0.0f);
        vertices[i].tangent = hasVertexTangents
            ? toGlm(tangents[i]) : glm::vec3(1.0f, 0.0f, 0.0f);
        vertices[i].tangentSign = hasVertexTangentSigns ? tangentSigns[i] : 1.0f;
        vertices[i].uv = hasVertexUvs
            ? glm::vec2(uvs[i][0], uvs[i][1]) : glm::vec2(0.0f);
        const glm::vec3 color = hasVertexColors
            ? toGlm(displayColors[i]) : glm::vec3(1.0f);
        const float opacity = hasVertexOpacities ? displayOpacities[i] : 1.0f;
        vertices[i].color = nr::vertex_color::packLinear(
            glm::vec4(color, opacity));
    }
    std::vector<uint32_t> triangleIndices;
    std::vector<Face> faces;
    VtIntArray faceMaterialIndices;
    getAttr(mesh.GetPrim(), "nr:faceMaterialIndices", &faceMaterialIndices);
    size_t cursor = 0;
    for (int count : counts) {
        if (count < 3 || cursor + static_cast<size_t>(count) > indices.size()) break;
        for (int i = 1; i + 1 < count; ++i) {
            triangleIndices.push_back(static_cast<uint32_t>(indices[cursor]));
            triangleIndices.push_back(static_cast<uint32_t>(indices[cursor + i]));
            triangleIndices.push_back(static_cast<uint32_t>(indices[cursor + i + 1]));
            const size_t faceIndex = faces.size();
            faces.push_back({faceIndex < faceMaterialIndices.size()
                ? faceMaterialIndices[faceIndex] : 0});
        }
        cursor += static_cast<size_t>(count);
    }
    VtStringArray materialPaths;
    getAttr(mesh.GetPrim(), "nr:materialPaths", &materialPaths);
    std::vector<MaterialRef> materialRefs;
    for (const std::string& path : materialPaths)
        materialRefs.push_back(loadMaterial(stage, path, scene, materials));
    if (materialRefs.empty()) materialRefs.push_back(scene.addMaterial(nr::materialx::defaultMaterial()));
    MeshGeometry geometry;
    geometry.vertices = nr::rstd::vector<Vertex>(vertices.begin(), vertices.end());
    geometry.indices = nr::rstd::vector<uint32_t>(triangleIndices.begin(), triangleIndices.end());
    geometry.faces = nr::rstd::vector<Face>(faces.begin(), faces.end());
    auto asset = scene.add(MeshAsset(scene, name, std::move(geometry), materialRefs));
    scene.add(std::make_unique<MeshInstance>(scene, name, asset, transform));
}

void readObject(Scene& scene, const UsdPrim& prim, const UsdStageRefPtr& stage,
    MaterialTable& materials)
{
    const Transform transform = readTransform(prim);
    const std::string name = prim.GetName().GetString();
    if (prim.IsA<pxr::UsdGeomMesh>()) {
        readMesh(scene, pxr::UsdGeomMesh(prim), name, transform, stage, materials);
        return;
    }
    if (prim.IsA<pxr::UsdGeomCamera>()) {
        pxr::UsdGeomCamera camera(prim);
        nr::sceneio::CameraFile file{};
        TfToken projection;
        camera.GetProjectionAttr().Get(&projection);
        file.projection = projection.GetString();
        float focalLengthMm = 50.0f;
        if (camera.GetFocalLengthAttr().Get(&focalLengthMm))
            file.focal_length_mm = focalLengthMm;
        float focusMm = file.focus_distance_cm * 10.0f;
        camera.GetFocusDistanceAttr().Get(&focusMm);
        file.focus_distance_cm = focusMm / 10.0f;
        camera.GetHorizontalApertureAttr().Get(&file.sensor_width_mm);
        camera.GetVerticalApertureAttr().Get(&file.sensor_height_mm);
        getAttr(prim, "nr:active", &file.active);
        getAttr(prim, "nr:exposure", &file.exposure);
        pxr::GfVec2i resolution(1280, 720);
        if (getAttr(prim, "nr:resolution", &resolution))
            file.resolution = {static_cast<uint32_t>(std::max(resolution[0], 1)), static_cast<uint32_t>(std::max(resolution[1], 1))};
        std::string sensorType;
        if (getAttr(prim, "nr:sensorType", &sensorType)) file.sensor_type = sensorType;
        // Reuse the mature legacy camera construction path by authoring the
        // small in-memory record it already understands.
        std::unique_ptr<Sensor> sensor;
        if (file.sensor_type == "scatter_psf") sensor = std::make_unique<ScatterPsfSensor>();
        else if (file.sensor_type == "gather_psf") sensor = std::make_unique<GatherPsfSensor>();
        else sensor = std::make_unique<RectangularSensor>();
        std::unique_ptr<Camera> cameraObject;
        if (file.projection == "orthographic") cameraObject = std::make_unique<OrthographicCamera>(std::move(sensor));
        else cameraObject = std::make_unique<PerspectiveCamera>(std::move(sensor));
        cameraObject->setFocalLengthMm(file.focal_length_mm.value_or(50.0f));
        cameraObject->setFocusDistanceCm(file.focus_distance_cm);
        cameraObject->exposure = file.exposure;
        cameraObject->getSensor().setDimensionsMm(file.sensor_width_mm, file.sensor_height_mm);
        cameraObject->getSensor().setResolution(file.resolution[0], file.resolution[1]);
        auto instance = std::make_unique<CameraInstance>(std::move(cameraObject), name, transform);
        CameraInstance* result = instance.get();
        scene.add(std::move(instance));
        if (file.active) scene.setActiveCamera(result);
        return;
    }
    int lightType = -1;
    getAttr(prim, "nr:lightType", &lightType);
    if (lightType < 0) {
        if (prim.IsA<pxr::UsdLuxDistantLight>()) lightType = LightInstance::TypeDirectional;
        else if (prim.IsA<pxr::UsdLuxRectLight>()) lightType = LightInstance::TypeRect;
        else if (prim.IsA<pxr::UsdLuxSphereLight>()) lightType = LightInstance::TypePoint;
    }
    if (lightType >= 0) {
        auto light = std::make_unique<LightInstance>(scene, name, transform, lightType);
        glm::vec3 color(1.0f); float intensity = 1.0f;
        GfVec3f usdColor;
        if (getAttr(prim, "nr:color", &usdColor)) color = toGlm(usdColor);
        else if (const pxr::UsdLuxLightAPI usdLight(prim); usdLight.GetColorAttr().Get(&usdColor)) color = toGlm(usdColor);
        if (!getAttr(prim, "nr:intensity", &intensity))
            pxr::UsdLuxLightAPI(prim).GetIntensityAttr().Get(&intensity);
        light->setPhotometry(color, intensity);
        if (lightType == LightInstance::TypePoint || lightType == LightInstance::TypeSpot) {
            float radius = 0.0f;
            if (pxr::UsdLuxSphereLight(prim).GetRadiusAttr().Get(&radius))
                lightType == LightInstance::TypePoint ? light->setPointRadius(radius) : light->setSpotRadius(radius);
        } else if (lightType == LightInstance::TypeRect) {
            float width = 1.0f, height = 1.0f;
            const pxr::UsdLuxRectLight usdRect(prim);
            usdRect.GetWidthAttr().Get(&width);
            usdRect.GetHeightAttr().Get(&height);
            auto& rect = std::get<RectLight>(light->getLightData());
            rect.width = width;
            rect.height = height;
        } else if (lightType == LightInstance::TypeDirectional) {
            float angle = 0.53f;
            if (pxr::UsdLuxDistantLight(prim).GetAngleAttr().Get(&angle))
                light->setDirectionalSoftAngle(angle);
        }
        light->commitLightChanges();
        scene.add(std::move(light));
        return;
    }
    std::string sourceType, sourcePath;
    getAttr(prim, "nr:sourceType", &sourceType);
    getAttr(prim, "nr:sourcePath", &sourcePath);
    if (sourceType == "gaussian" && !sourcePath.empty()) {
        SceneImporter::ImportGaussianScene(scene, sourcePath);
        if (SceneObject* root = scene.getActiveObject()) root->setLocalTransform(transform);
    }
}

} // namespace

namespace nr::sceneio {

bool isUsdFile(const std::string& filepath)
{
    const std::string extension = lower(std::filesystem::path(filepath).extension().string());
    return extension == ".usd" || extension == ".usda" || extension == ".usdc" || extension == ".usdz";
}

void writeUsd(const Scene& scene, const std::string& filepath)
{
    const std::filesystem::path usdPath(filepath);
    const UsdStageRefPtr stage = pxr::UsdStage::CreateNew(filepath);
    if (!stage) throw std::runtime_error("Failed to create USD stage: " + filepath);
    const pxr::UsdGeomXform root = pxr::UsdGeomXform::Define(stage, SdfPath(kSceneRoot));
    const pxr::UsdGeomXform materialsRoot = pxr::UsdGeomXform::Define(stage, SdfPath(kMaterialsRoot));
    (void)materialsRoot;
    // Author the layer metadata directly.  This avoids a Blender USD runtime
    // quirk in UsdStage::SetDefaultPrim while still making the scene discoverable
    // through the standard USD default-prim API used by Hydra clients.
    stage->GetRootLayer()->SetDefaultPrim(TfToken("NoorRayScene"));

    setAttr(root.GetPrim(), "nr:environmentColor", pxr::SdfValueTypeNames->Color3f,
        toUsd(scene.getEnvironment().color));
    setAttr(root.GetPrim(), "nr:lightingExposure", pxr::SdfValueTypeNames->Float,
        scene.getEnvironment().lightingExposure);
    setAttr(root.GetPrim(), "nr:visibleExposure", pxr::SdfValueTypeNames->Float,
        scene.getEnvironment().visibleExposure);
    setAttr(root.GetPrim(), "nr:maxSamples", pxr::SdfValueTypeNames->Int, scene.getRenderSettings().maxSamples);
    setAttr(root.GetPrim(), "nr:aovEnabled", pxr::SdfValueTypeNames->Bool, scene.getRenderSettings().aovEnabled);
    setAttr(root.GetPrim(), "nr:optixDenoiserEnabled", pxr::SdfValueTypeNames->Bool, scene.getRenderSettings().optixDenoiserEnabled);
    setAttr(root.GetPrim(), "nr:optixDenoiserMinSamples", pxr::SdfValueTypeNames->Int, scene.getRenderSettings().optixDenoiserMinSamples);
    setAttr(root.GetPrim(), "nr:indirectLightClamp", pxr::SdfValueTypeNames->Float, scene.getRenderSettings().indirectLightClamp);
    MaterialTable materialTable;
    for (const auto& object : scene.getRootObjects())
        writeObject(scene, object, stage, root.GetPath(), usdPath, materialTable, scene.getActiveCamera());
    if (!stage->GetRootLayer()->Save())
        throw std::runtime_error("Failed to save USD scene: " + filepath);
}

void readUsd(Scene& scene, const std::string& filepath)
{
    const UsdStageRefPtr stage = pxr::UsdStage::Open(filepath);
    if (!stage) throw std::runtime_error("Failed to open USD scene: " + filepath);
    UsdPrim root = stage->GetPrimAtPath(SdfPath(kSceneRoot));
    if (!root) root = stage->GetDefaultPrim();
    if (!root) root = stage->GetPseudoRoot();
    if (!root) throw std::runtime_error("USD scene has no /NoorRayScene root: " + filepath);
    scene.clear();
    if (GfVec3f color; getAttr(root, "nr:environmentColor", &color)) scene.getEnvironment().color = toGlm(color);
    getAttr(root, "nr:lightingExposure", &scene.getEnvironment().lightingExposure);
    getAttr(root, "nr:visibleExposure", &scene.getEnvironment().visibleExposure);
    getAttr(root, "nr:maxSamples", &scene.getRenderSettings().maxSamples);
    getAttr(root, "nr:aovEnabled", &scene.getRenderSettings().aovEnabled);
    getAttr(root, "nr:optixDenoiserEnabled", &scene.getRenderSettings().optixDenoiserEnabled);
    getAttr(root, "nr:optixDenoiserMinSamples", &scene.getRenderSettings().optixDenoiserMinSamples);
    getAttr(root, "nr:indirectLightClamp", &scene.getRenderSettings().indirectLightClamp);
    scene.getRenderSettings().indirectLightClamp = std::max(
        scene.getRenderSettings().indirectLightClamp, 0.0f);
    scene.getEnvironment().updateDerivedSettings();
    MaterialTable materials;
    for (const UsdPrim& prim : root.GetChildren())
        if (prim.GetPath() != SdfPath(kMaterialsRoot)) readObject(scene, prim, stage, materials);
    scene.reclaimUnusedResources();
}

} // namespace nr::sceneio
