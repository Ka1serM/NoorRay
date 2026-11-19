#include "NoorRay/Import/UsdImporter.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <span>
#include <tuple>
#include <unordered_map>

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/shapingAPI.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

#include "NoorRay/Core/Log.h"
#include "NoorRay/Import/MeshBuilder.h"
#include "NoorRay/Import/PreviewSurface.h"
#include "NoorRay/Materials/TextureCache.h"

namespace NoorRay::Import {

namespace {

Core::Transform toTransform(const pxr::GfMatrix4d& matrix) {
    //! USD matrices act on row vectors, so the transpose is the column-vector transform.
    Core::Transform transform;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 4; ++column)
            transform.rows[row][column] = static_cast<float>(matrix[column][row]);
    return transform;
}

struct CornerKey {
    std::uint32_t point = 0;
    std::uint32_t normal = 0;
    std::uint32_t uv = 0;
    std::uint32_t color = 0;
    bool operator==(const CornerKey&) const = default;
};

struct CornerKeyHash {
    std::size_t operator()(const CornerKey& key) const {
        return (static_cast<std::size_t>(key.point) * 0x9e3779b97f4a7c15ull) ^ (static_cast<std::size_t>(key.normal) << 21)
               ^ (static_cast<std::size_t>(key.uv) << 42) ^ key.color;
    }
};

//! A primvar flattened to one value index per face corner.
template <class T>
struct CornerValues {
    pxr::VtArray<T> values;
    std::vector<std::uint32_t> indices;

    bool isEmpty() const { return indices.empty(); }
};

template <class T>
CornerValues<T> readCorners(const pxr::UsdGeomPrimvar& primvar, const pxr::VtIntArray& faceCounts, const pxr::VtIntArray& faceIndices) {
    CornerValues<T> result;
    if (!primvar || !primvar.HasValue() || !primvar.Get(&result.values) || result.values.empty())
        return result;
    pxr::VtIntArray valueIndices;
    primvar.GetIndices(&valueIndices);
    const auto lookup = [&](const std::size_t index) {
        return static_cast<std::uint32_t>(valueIndices.empty() ? index : static_cast<std::size_t>(valueIndices[index]));
    };
    const pxr::TfToken interpolation = primvar.GetInterpolation();
    result.indices.resize(faceIndices.size());
    std::size_t corner = 0;
    for (std::size_t face = 0; face < faceCounts.size(); ++face) {
        for (int vertex = 0; vertex < faceCounts[face]; ++vertex, ++corner) {
            std::size_t index = 0;
            if (interpolation == pxr::UsdGeomTokens->faceVarying)
                index = corner;
            else if (interpolation == pxr::UsdGeomTokens->uniform)
                index = face;
            else if (interpolation == pxr::UsdGeomTokens->constant)
                index = 0;
            else
                index = static_cast<std::size_t>(faceIndices[corner]);
            result.indices[corner] = lookup(index);
        }
    }
    const std::size_t count = result.values.size();
    if (std::ranges::any_of(result.indices, [count](const std::uint32_t index) { return index >= count; }))
        result.indices.clear();
    return result;
}

class Importer {
public:
    Importer(const pxr::UsdStageRefPtr& stage, Scene::Scene& scene) : stage(stage), scene(scene), textureCache(scene) {
        const double metersPerUnit = pxr::UsdGeomGetStageMetersPerUnit(stage);
        const float scale = static_cast<float>(metersPerUnit > 0.0 ? metersPerUnit : 0.01);
        //! The scene is +Y up in metres; a Z-up stage turns -90 degrees about X.
        if (pxr::UsdGeomGetStageUpAxis(stage) == pxr::UsdGeomTokens->z) {
            correction.rows[1][1] = 0.0f;
            correction.rows[1][2] = 1.0f;
            correction.rows[2][1] = -1.0f;
            correction.rows[2][2] = 0.0f;
        }
        for (int row = 0; row < 3; ++row)
            for (int column = 0; column < 3; ++column)
                correction.rows[row][column] *= scale;
        unitScale = scale;
    }

    std::expected<void, Core::Error> run() {
        pxr::UsdGeomXformCache xformCache;
        for (const pxr::UsdPrim& prim : pxr::UsdPrimRange::Stage(stage, pxr::UsdTraverseInstanceProxies())) {
            if (!prim.IsActive())
                continue;
            const pxr::UsdGeomImageable imageable(prim);
            if (imageable && imageable.ComputeVisibility() == pxr::UsdGeomTokens->invisible)
                continue;
            const Core::Transform transform = correction * toTransform(xformCache.GetLocalToWorldTransform(prim));
            Scene::NodeObject object;
            if (prim.IsA<pxr::UsdGeomMesh>()) {
                std::expected<Scene::MeshHandle, Core::Error> mesh = getMesh(prim);
                if (!mesh)
                    return std::unexpected(std::move(mesh.error()));
                if (mesh->isValid())
                    object = scene.addInstance(
                        {transform, transform, *mesh, Scene::InstanceFlags::None, static_cast<std::uint32_t>(scene.getInstances().size())});
            } else if (prim.IsA<pxr::UsdGeomCamera>()) {
                object = addCamera(pxr::UsdGeomCamera(prim), transform);
            } else if (prim.HasAPI<pxr::UsdLuxLightAPI>()) {
                object = addLight(prim, transform);
                if (object.index() == 0)
                    continue;
            } else if (!prim.IsA<pxr::UsdGeomXformable>()) {
                continue;
            }
            addNode(prim, object, transform);
        }
        return {};
    }

private:
    //! Cameras and lights drop the scale of their prim, so a node takes the transform its object really has.
    void addNode(const pxr::UsdPrim& prim, const Scene::NodeObject& object, Core::Transform world) {
        if (const auto* light = std::get_if<Scene::LightHandle>(&object))
            world = scene.find(*light)->transform;
        else if (const auto* camera = std::get_if<Scene::CameraHandle>(&object))
            world = scene.find(*camera)->transform;
        const Scene::NodeHandle parent = findParentNode(prim);
        const Scene::Node* parentNode = scene.find(parent);
        const Core::Transform local = parentNode ? parentNode->worldTransform.inverse() * world : world;
        nodes.emplace(prim.GetPath(), scene.addNode({prim.GetName().GetString(), local, parent, object}));
    }

    Scene::NodeHandle findParentNode(const pxr::UsdPrim& prim) const {
        for (pxr::UsdPrim ancestor = prim.GetParent(); ancestor; ancestor = ancestor.GetParent())
            if (const auto found = nodes.find(ancestor.GetPath()); found != nodes.end())
                return found->second;
        return {};
    }

    std::optional<Scene::MaterialTexture> resolveTexture(const std::string& path, const Materials::ColorSpace colorSpace) {
        std::expected<Scene::TextureHandle, Core::Error> handle = textureCache.load(path, colorSpace);
        if (!handle) {
            Core::Log::warning("USD texture '{}' is skipped: {}", path, handle.error().message);
            return std::nullopt;
        }
        Scene::MaterialTexture texture;
        texture.texture = *handle;
        return texture;
    }

    Scene::MaterialHandle getMaterial(const pxr::UsdShadeMaterial& material, const bool vertexColors, const Core::float3 displayColor) {
        const std::string path = material ? material.GetPath().GetString() : std::string();
        const auto key = std::make_tuple(path, vertexColors, displayColor.x, displayColor.y, displayColor.z);
        if (const auto found = materials.find(key); found != materials.end())
            return found->second;
        std::optional<Scene::Material> converted;
        if (material) {
            converted = PreviewSurface::convert(
                material, [this](const std::string& file, const Materials::ColorSpace colorSpace) { return resolveTexture(file, colorSpace); });
            if (!converted)
                Core::Log::warning("USD material '{}' has no UsdPreviewSurface; it is imported as grey", path);
        }
        Scene::Material result = converted.value_or(Scene::Material{});
        if (!converted) {
            result.name = path.empty() ? "displayColor" : path;
            result.baseColor = displayColor;
            result.useVertexColor = vertexColors;
        }
        const Scene::MaterialHandle handle = scene.addMaterial(std::move(result));
        materials.emplace(key, handle);
        return handle;
    }

    std::expected<Scene::MeshHandle, Core::Error> getMesh(const pxr::UsdPrim& prim) {
        const pxr::UsdPrim source = prim.IsInstanceProxy() ? prim.GetPrimInPrototype() : prim;
        const std::string key = source.GetPath().GetString();
        if (const auto found = meshes.find(key); found != meshes.end())
            return found->second;
        std::expected<Scene::MeshHandle, Core::Error> mesh = readMesh(prim);
        if (mesh)
            meshes.emplace(key, *mesh);
        return mesh;
    }

    std::expected<Scene::MeshHandle, Core::Error> readMesh(const pxr::UsdPrim& prim) {
        const pxr::UsdGeomMesh mesh(prim);
        pxr::VtVec3fArray points;
        pxr::VtIntArray faceCounts;
        pxr::VtIntArray faceIndices;
        mesh.GetPointsAttr().Get(&points);
        mesh.GetFaceVertexCountsAttr().Get(&faceCounts);
        mesh.GetFaceVertexIndicesAttr().Get(&faceIndices);
        if (points.empty() || faceCounts.empty())
            return Scene::MeshHandle{};
        std::size_t cornerCount = 0;
        for (const int count : faceCounts)
            cornerCount += static_cast<std::size_t>(std::max(count, 0));
        if (cornerCount != faceIndices.size()
            || std::ranges::any_of(faceIndices, [&](const int index) { return index < 0 || static_cast<std::size_t>(index) >= points.size(); }))
            return Core::fail(Core::ErrorCode::InvalidFile, std::format("USD mesh '{}' has invalid face indices", prim.GetPath().GetString()));

        const pxr::UsdGeomPrimvarsAPI primvars(prim);
        CornerValues<pxr::GfVec3f> normals;
        if (pxr::UsdGeomPrimvar primvar = primvars.GetPrimvar(pxr::TfToken("normals")); primvar && primvar.HasValue()) {
            normals = readCorners<pxr::GfVec3f>(primvar, faceCounts, faceIndices);
        } else {
            pxr::TfToken scheme;
            mesh.GetSubdivisionSchemeAttr().Get(&scheme);
            pxr::UsdAttribute attribute = mesh.GetNormalsAttr();
            if (scheme == pxr::UsdGeomTokens->none && attribute.HasValue()) {
                attribute.Get(&normals.values);
                const pxr::TfToken interpolation = mesh.GetNormalsInterpolation();
                normals.indices.resize(cornerCount);
                std::size_t corner = 0;
                for (std::size_t face = 0; face < faceCounts.size(); ++face)
                    for (int vertex = 0; vertex < faceCounts[face]; ++vertex, ++corner)
                        normals.indices[corner] = static_cast<std::uint32_t>(interpolation == pxr::UsdGeomTokens->faceVarying ? corner
                                                                             : interpolation == pxr::UsdGeomTokens->uniform   ? face
                                                                                                                              : faceIndices[corner]);
                const std::size_t count = normals.values.size();
                if (std::ranges::any_of(normals.indices, [count](const std::uint32_t index) { return index >= count; }))
                    normals.indices.clear();
            }
        }
        pxr::UsdGeomPrimvar uvPrimvar = primvars.GetPrimvar(pxr::TfToken("st"));
        if (!uvPrimvar || !uvPrimvar.HasValue())
            for (const pxr::UsdGeomPrimvar& candidate : primvars.GetPrimvarsWithValues())
                if (candidate.GetTypeName() == pxr::SdfValueTypeNames->TexCoord2fArray || candidate.GetTypeName() == pxr::SdfValueTypeNames->Float2Array) {
                    uvPrimvar = candidate;
                    break;
                }
        const CornerValues<pxr::GfVec2f> uvs = readCorners<pxr::GfVec2f>(uvPrimvar, faceCounts, faceIndices);
        const CornerValues<pxr::GfVec3f> colors = readCorners<pxr::GfVec3f>(primvars.GetPrimvar(pxr::TfToken("displayColor")), faceCounts, faceIndices);
        const bool constantColor = colors.isEmpty() || colors.values.size() == 1;
        const Core::float3 displayColor =
            !colors.isEmpty() ? Core::float3{colors.values[0][0], colors.values[0][1], colors.values[0][2]} : Core::float3{0.8f, 0.8f, 0.8f};

        const pxr::UsdShadeMaterialBindingAPI binding(prim);
        const pxr::UsdShadeMaterial meshMaterial = binding.ComputeBoundMaterial();
        std::vector<Scene::MaterialHandle> faceMaterials(faceCounts.size(), getMaterial(meshMaterial, !constantColor, displayColor));
        for (const pxr::UsdGeomSubset& subset : binding.GetMaterialBindSubsets()) {
            pxr::VtIntArray faces;
            subset.GetIndicesAttr().Get(&faces);
            const pxr::UsdShadeMaterial subsetMaterial = pxr::UsdShadeMaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial();
            const Scene::MaterialHandle handle = getMaterial(subsetMaterial, !constantColor, displayColor);
            for (const int face : faces)
                if (face >= 0 && static_cast<std::size_t>(face) < faceMaterials.size())
                    faceMaterials[face] = handle;
        }

        pxr::TfToken orientation;
        mesh.GetOrientationAttr().Get(&orientation);
        const bool leftHanded = orientation == pxr::UsdGeomTokens->leftHanded;

        MeshData data;
        data.name = prim.GetPath().GetString();
        std::vector<std::size_t> faceStarts(faceCounts.size());
        for (std::size_t face = 0, start = 0; face < faceCounts.size(); start += static_cast<std::size_t>(faceCounts[face]), ++face)
            faceStarts[face] = start;
        std::vector<std::uint32_t> order(faceCounts.size());
        for (std::uint32_t face = 0; face < order.size(); ++face)
            order[face] = face;
        std::ranges::stable_sort(order, [&](const std::uint32_t a, const std::uint32_t b) { return faceMaterials[a].index < faceMaterials[b].index; });

        std::unordered_map<CornerKey, std::uint32_t, CornerKeyHash> vertices;
        const auto vertexOf = [&](const std::size_t corner) {
            const CornerKey key{static_cast<std::uint32_t>(faceIndices[corner]), normals.isEmpty() ? 0 : normals.indices[corner],
                                uvs.isEmpty() ? 0 : uvs.indices[corner], constantColor ? 0 : colors.indices[corner]};
            const auto [found, inserted] = vertices.try_emplace(key, static_cast<std::uint32_t>(data.positions.size()));
            if (!inserted)
                return found->second;
            const pxr::GfVec3f& point = points[key.point];
            data.positions.push_back({point[0], point[1], point[2]});
            if (!normals.isEmpty()) {
                const pxr::GfVec3f& normal = normals.values[key.normal];
                data.normals.push_back({normal[0], normal[1], normal[2]});
            }
            if (!uvs.isEmpty())
                data.uvs[0].push_back({uvs.values[key.uv][0], uvs.values[key.uv][1]});
            if (!constantColor) {
                const pxr::GfVec3f& color = colors.values[key.color];
                data.colors.push_back({color[0], color[1], color[2], 1.0f});
            }
            return found->second;
        };
        for (const std::uint32_t face : order) {
            const std::uint32_t firstIndex = static_cast<std::uint32_t>(data.indices.size());
            const std::size_t start = faceStarts[face];
            for (int corner = 2; corner < faceCounts[face]; ++corner) {
                const std::uint32_t a = vertexOf(start);
                const std::uint32_t b = vertexOf(start + corner - 1);
                const std::uint32_t c = vertexOf(start + corner);
                if (leftHanded)
                    data.indices.insert(data.indices.end(), {a, c, b});
                else
                    data.indices.insert(data.indices.end(), {a, b, c});
            }
            data.appendSubset(faceMaterials[face], firstIndex, static_cast<std::uint32_t>(data.indices.size()) - firstIndex);
        }
        return scene.addMesh(MeshBuilder::build(std::move(data)));
    }

    Scene::CameraHandle addCamera(const pxr::UsdGeomCamera& source, const Core::Transform& transform) {
        const pxr::GfCamera camera = source.GetCamera(pxr::UsdTimeCode::Default());
        Scene::Camera result;
        result.name = source.GetPath().GetString();
        //! The transform may scale; the camera keeps only its rotation and position.
        Core::Transform frame = transform;
        for (int column = 0; column < 3; ++column) {
            const Core::float3 axis = Core::normalize(frame.getColumn(column));
            frame.rows[0][column] = axis.x;
            frame.rows[1][column] = axis.y;
            frame.rows[2][column] = axis.z;
        }
        result.transform = frame;
        result.previousTransform = frame;
        //! Apertures and focal length are in tenths of a scene unit by the UsdGeomCamera convention, so millimetres for centimetre stages.
        const float tenth = unitScale * 0.1f;
        result.sensor.width = static_cast<float>(camera.GetHorizontalAperture()) * tenth;
        result.sensor.height = static_cast<float>(camera.GetVerticalAperture()) * tenth;
        result.sensor.offset = {static_cast<float>(camera.GetHorizontalApertureOffset()) * tenth,
                                static_cast<float>(camera.GetVerticalApertureOffset()) * tenth};
        const float focalLength = static_cast<float>(camera.GetFocalLength()) * tenth;
        if (camera.GetProjection() == pxr::GfCamera::Orthographic) {
            result.type = Scene::CameraType::Orthographic;
            result.orthographicHeight = result.sensor.height;
        } else {
            result.verticalFov = 2.0f * std::atan(0.5f * result.sensor.height / std::max(focalLength, 1e-6f));
        }
        result.fStop = camera.GetFStop();
        if (camera.GetFocusDistance() > 0.0f)
            result.focusDistance = camera.GetFocusDistance() * unitScale;
        const pxr::GfRange1f clipping = camera.GetClippingRange();
        result.near = clipping.GetMin() * unitScale;
        result.far = clipping.GetMax() * unitScale;
        const Scene::CameraHandle handle = scene.addCamera(std::move(result));
        if (!scene.findActiveCamera())
            scene.setActiveCamera(handle);
        return handle;
    }

    Scene::NodeObject addLight(const pxr::UsdPrim& prim, const Core::Transform& transform) {
        const pxr::UsdLuxLightAPI api(prim);
        float intensity = 1.0f;
        float exposure = 0.0f;
        pxr::GfVec3f color(1.0f);
        bool normalize = false;
        api.GetIntensityAttr().Get(&intensity);
        api.GetExposureAttr().Get(&exposure);
        api.GetColorAttr().Get(&color);
        api.GetNormalizeAttr().Get(&normalize);
        //! UsdLux intensity is a radiance in nits; the scene's power is that radiance over the emitting area times pi.
        const float radiance = intensity * std::exp2(exposure);
        Scene::Light light;
        light.name = prim.GetPath().GetString();
        light.color = {color[0], color[1], color[2]};
        Core::Transform frame = transform;
        float scale = 0.0f;
        for (int column = 0; column < 3; ++column) {
            const Core::float3 axis = frame.getColumn(column);
            scale = std::max(scale, Core::length(axis));
            const Core::float3 unit = Core::normalize(axis);
            frame.rows[0][column] = unit.x;
            frame.rows[1][column] = unit.y;
            frame.rows[2][column] = unit.z;
        }
        light.transform = frame;
        if (prim.IsA<pxr::UsdLuxDomeLight>()) {
            Scene::Environment environment;
            environment.color = light.color;
            environment.strength = radiance;
            pxr::SdfAssetPath file;
            pxr::UsdLuxDomeLight(prim).GetTextureFileAttr().Get(&file);
            const std::string path = file.GetResolvedPath().empty() ? file.GetAssetPath() : file.GetResolvedPath();
            if (!path.empty()) {
                if (std::expected<Scene::TextureHandle, Core::Error> texture = textureCache.load(path, Materials::ColorSpace::Linear))
                    environment.texture = *texture;
                else
                    Core::Log::warning("USD dome texture '{}' is skipped: {}", path, texture.error().message);
            }
            const Core::float3 forward = frame.transformVector({0.0f, 0.0f, -1.0f});
            environment.rotation = std::atan2(forward.x, -forward.z);
            scene.setEnvironment(environment);
            return {};
        }
        if (prim.IsA<pxr::UsdLuxDistantLight>()) {
            float angle = 0.53f;
            pxr::UsdLuxDistantLight(prim).GetAngleAttr().Get(&angle);
            light.type = Scene::LightType::Directional;
            light.angle = angle * Core::pi / 180.0f;
            light.power = radiance;
            return scene.addLight(std::move(light));
        }
        float area = 1.0f;
        if (prim.IsA<pxr::UsdLuxSphereLight>()) {
            float radius = 0.5f;
            bool treatAsPoint = false;
            pxr::UsdLuxSphereLight(prim).GetRadiusAttr().Get(&radius);
            pxr::UsdLuxSphereLight(prim).GetTreatAsPointAttr().Get(&treatAsPoint);
            light.radius = treatAsPoint ? 0.0f : radius * scale;
            light.type = light.radius > 0.0f ? Scene::LightType::Sphere : Scene::LightType::Point;
            area = 4.0f * Core::pi * radius * radius * scale * scale;
        } else if (prim.IsA<pxr::UsdLuxRectLight>()) {
            float width = 1.0f;
            float height = 1.0f;
            pxr::UsdLuxRectLight(prim).GetWidthAttr().Get(&width);
            pxr::UsdLuxRectLight(prim).GetHeightAttr().Get(&height);
            light.type = Scene::LightType::Rect;
            light.width = width * scale;
            light.height = height * scale;
            area = light.width * light.height;
        } else if (prim.IsA<pxr::UsdLuxDiskLight>()) {
            float radius = 0.5f;
            pxr::UsdLuxDiskLight(prim).GetRadiusAttr().Get(&radius);
            light.type = Scene::LightType::Disk;
            light.radius = radius * scale;
            area = Core::pi * light.radius * light.radius;
        } else {
            Core::Log::warning("USD light '{}' has an unsupported type and is skipped", light.name);
            return {};
        }
        light.power = radiance * Core::pi * (normalize ? 1.0f : area);
        if (prim.HasAPI<pxr::UsdLuxShapingAPI>()) {
            const pxr::UsdLuxShapingAPI shaping(prim);
            float coneAngle = 90.0f;
            float softness = 0.0f;
            shaping.GetShapingConeAngleAttr().Get(&coneAngle);
            shaping.GetShapingConeSoftnessAttr().Get(&softness);
            if (coneAngle < 90.0f && light.type == Scene::LightType::Point) {
                light.type = Scene::LightType::Spot;
                light.spotAngle = 2.0f * coneAngle * Core::pi / 180.0f;
                light.spotBlend = std::clamp(softness, 0.0f, 1.0f);
            }
        }
        return scene.addLight(std::move(light));
    }

    pxr::UsdStageRefPtr stage;
    Scene::Scene& scene;
    Materials::TextureCache textureCache;
    Core::Transform correction;
    float unitScale = 1.0f;
    std::unordered_map<std::string, Scene::MeshHandle> meshes;
    std::map<std::tuple<std::string, bool, float, float, float>, Scene::MaterialHandle> materials;
    std::unordered_map<pxr::SdfPath, Scene::NodeHandle, pxr::SdfPath::Hash> nodes;
};

}

std::expected<void, Core::Error> UsdImporter::import(const std::filesystem::path& path, Scene::Scene& scene) {
    const pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(path.string());
    if (!stage)
        return Core::fail(Core::ErrorCode::InvalidFile, std::format("cannot open USD stage '{}'", path.string()));
    return Importer(stage, scene).run();
}

}
