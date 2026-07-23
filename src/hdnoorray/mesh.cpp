#include "mesh.h"

#include "instancer.h"
#include "renderParam.h"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/imaging/hd/changeTracker.h>
#include <pxr/imaging/hd/renderIndex.h>
#include <pxr/imaging/hd/sceneDelegate.h>
#include <pxr/imaging/hd/tokens.h>

#include <algorithm>
#include <mutex>
#include <vector>

#include <glm/geometric.hpp>

#include "Mesh/Assets/MeshAsset.h"
#include "Mesh/Transform.h"
#include "Scene/MeshInstance.h"
#include "Scene/Scene.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace
{
glm::mat4 ToGlm(const GfMatrix4d& value)
{
    glm::mat4 result(1.0f);
    for (int row = 0; row < 4; ++row)
        for (int column = 0; column < 4; ++column)
            result[column][row] = static_cast<float>(value[column][row]);
    return result;
}

std::vector<glm::vec3> Vec3Values(const VtValue& value)
{
    std::vector<glm::vec3> result;
    if (value.IsHolding<VtVec3fArray>()) {
        for (const GfVec3f& v : value.UncheckedGet<VtVec3fArray>())
            result.emplace_back(v[0], v[1], v[2]);
    } else if (value.IsHolding<VtVec3dArray>()) {
        for (const GfVec3d& v : value.UncheckedGet<VtVec3dArray>())
            result.emplace_back(v[0], v[1], v[2]);
    }
    return result;
}

std::vector<glm::vec2> Vec2Values(const VtValue& value)
{
    std::vector<glm::vec2> result;
    if (value.IsHolding<VtVec2fArray>()) {
        for (const GfVec2f& v : value.UncheckedGet<VtVec2fArray>())
            result.emplace_back(v[0], v[1]);
    }
    return result;
}

HdInterpolation FindInterpolation(
    HdSceneDelegate* delegate, const SdfPath& id, const TfToken& name)
{
    for (const HdInterpolation interpolation : {
             HdInterpolationFaceVarying, HdInterpolationVertex,
             HdInterpolationVarying, HdInterpolationUniform,
             HdInterpolationConstant}) {
        for (const HdPrimvarDescriptor& descriptor :
             delegate->GetPrimvarDescriptors(id, interpolation))
            if (descriptor.name == name)
                return interpolation;
    }
    return HdInterpolationConstant;
}

template<typename T>
T PrimvarAt(
    const std::vector<T>& values, const HdInterpolation interpolation,
    const int pointIndex, const size_t cornerIndex, const size_t faceIndex,
    const T fallback)
{
    size_t index = 0;
    if (interpolation == HdInterpolationFaceVarying)
        index = cornerIndex;
    else if (interpolation == HdInterpolationVertex
        || interpolation == HdInterpolationVarying)
        index = pointIndex < 0 ? values.size() : static_cast<size_t>(pointIndex);
    else if (interpolation == HdInterpolationUniform)
        index = faceIndex;
    return index < values.size() ? values[index] : fallback;
}

void BuildTriangleMesh(
    HdSceneDelegate* delegate, const SdfPath& id,
    const HdMeshTopology& topology, const std::vector<glm::vec3>& positions,
    std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
    std::vector<Face>& faces)
{
    const std::vector<glm::vec3> normals =
        Vec3Values(delegate->Get(id, HdTokens->normals));
    const std::vector<glm::vec2> uvs =
        Vec2Values(delegate->Get(id, TfToken("st")));
    const HdInterpolation normalInterpolation =
        FindInterpolation(delegate, id, HdTokens->normals);
    const HdInterpolation uvInterpolation =
        FindInterpolation(delegate, id, TfToken("st"));

    const VtIntArray& counts = topology.GetFaceVertexCounts();
    const VtIntArray& sourceIndices = topology.GetFaceVertexIndices();
    const bool flip = topology.GetOrientation() == HdTokens->leftHanded;
    size_t sourceOffset = 0;
    for (size_t faceIndex = 0; faceIndex < counts.size(); ++faceIndex) {
        const int count = counts[faceIndex];
        if (count < 3 || sourceOffset + static_cast<size_t>(count) > sourceIndices.size()) {
            sourceOffset += std::max(count, 0);
            continue;
        }
        for (int fan = 1; fan + 1 < count; ++fan) {
            int localCorners[3] = {0, fan, fan + 1};
            if (flip)
                std::swap(localCorners[1], localCorners[2]);
            int pointIndices[3]{};
            bool valid = true;
            for (int c = 0; c < 3; ++c) {
                pointIndices[c] = sourceIndices[sourceOffset + localCorners[c]];
                valid &= pointIndices[c] >= 0
                    && static_cast<size_t>(pointIndices[c]) < positions.size();
            }
            if (!valid)
                continue;
            const glm::vec3 geometricNormal = glm::normalize(glm::cross(
                positions[pointIndices[1]] - positions[pointIndices[0]],
                positions[pointIndices[2]] - positions[pointIndices[0]]));
            Vertex triangle[3]{};
            for (int c = 0; c < 3; ++c) {
                const size_t cornerIndex = sourceOffset + localCorners[c];
                triangle[c].position = positions[pointIndices[c]];
                triangle[c].normal = PrimvarAt(
                    normals, normalInterpolation, pointIndices[c], cornerIndex,
                    faceIndex, geometricNormal);
                if (glm::length(triangle[c].normal) > 0.0f)
                    triangle[c].normal = glm::normalize(triangle[c].normal);
                triangle[c].uv = PrimvarAt(
                    uvs, uvInterpolation, pointIndices[c], cornerIndex,
                    faceIndex, glm::vec2(0.0f));
                triangle[c].tangentSign = 1.0f;
            }
            const glm::vec3 edge1 = triangle[1].position - triangle[0].position;
            const glm::vec3 edge2 = triangle[2].position - triangle[0].position;
            const glm::vec2 duv1 = triangle[1].uv - triangle[0].uv;
            const glm::vec2 duv2 = triangle[2].uv - triangle[0].uv;
            const float determinant = duv1.x * duv2.y - duv1.y * duv2.x;
            const glm::vec3 tangent = std::abs(determinant) > 1e-8f
                ? glm::normalize((edge1 * duv2.y - edge2 * duv1.y) / determinant)
                : glm::normalize(edge1);
            for (Vertex& vertex : triangle) {
                vertex.tangent = tangent;
                indices.push_back(static_cast<uint32_t>(vertices.size()));
                vertices.push_back(vertex);
            }
            faces.push_back({0});
        }
        sourceOffset += static_cast<size_t>(count);
    }
}
}

HdNoorRayMesh::HdNoorRayMesh(const SdfPath& id)
    : HdMesh(id)
{
}

HdNoorRayMesh::~HdNoorRayMesh() = default;

HdDirtyBits HdNoorRayMesh::GetInitialDirtyBitsMask() const
{
    return HdChangeTracker::InitRepr | HdChangeTracker::DirtyPoints
        | HdChangeTracker::DirtyTopology | HdChangeTracker::DirtyTransform
        | HdChangeTracker::DirtyVisibility | HdChangeTracker::DirtyPrimvar
        | HdChangeTracker::DirtyMaterialId | HdChangeTracker::DirtyInstancer;
}

HdDirtyBits HdNoorRayMesh::_PropagateDirtyBits(const HdDirtyBits bits) const
{
    return bits;
}

void HdNoorRayMesh::_InitRepr(const TfToken& reprToken, HdDirtyBits*)
{
    const auto found = std::ranges::find_if(
        _reprs, [&reprToken](const auto& repr) { return repr.first == reprToken; });
    if (found == _reprs.end())
        _reprs.emplace_back(reprToken, HdReprSharedPtr());
}

void HdNoorRayMesh::Sync(
    HdSceneDelegate* delegate, HdRenderParam* renderParam,
    HdDirtyBits* dirtyBits, const TfToken&)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    Scene& scene = param.session.scene;
    _UpdateInstancer(delegate, dirtyBits);
    const SdfPath requestedMaterialId = delegate->GetMaterialId(GetId());
    SetMaterialId(requestedMaterialId);

    const bool geometryDirty =
        (*dirtyBits & (HdChangeTracker::DirtyPoints
            | HdChangeTracker::DirtyTopology
            | HdChangeTracker::DirtyPrimvar)) != 0;
    if (geometryDirty) {
        const std::vector<glm::vec3> positions =
            Vec3Values(delegate->Get(GetId(), HdTokens->points));
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        std::vector<Face> faces;
        BuildTriangleMesh(
            delegate, GetId(), GetMeshTopology(delegate), positions,
            vertices, indices, faces);
        if (!vertices.empty() && !indices.empty()) {
            Material material;
            material.albedo = glm::vec3(0.8f);
            material.roughness = 0.5f;
            if (meshIndex_ == ~0u) {
                meshIndex_ = scene.add(MeshAsset(
                    scene, GetId().GetString(), vertices, indices, faces,
                    {material}));
            } else {
                scene.getMeshAsset(meshIndex_).replaceGeometry(
                    vertices, indices, faces);
            }
        }
    }

    if (meshIndex_ != ~0u && boundMaterialId_ != requestedMaterialId) {
        param.UnbindMaterial(boundMaterialId_, meshIndex_);
        boundMaterialId_ = requestedMaterialId;
        param.BindMaterial(boundMaterialId_, meshIndex_);
    }

    const bool instancesDirty = objectIds_.empty()
        || (*dirtyBits & (HdChangeTracker::DirtyTransform
            | HdChangeTracker::DirtyInstancer)) != 0;
    if (meshIndex_ != ~0u && instancesDirty) {
        VtMatrix4dArray transforms;
        if (GetInstancerId().IsEmpty()) {
            transforms.push_back(delegate->GetTransform(GetId()));
        } else {
            HdInstancer::_SyncInstancerAndParents(
                delegate->GetRenderIndex(), GetInstancerId());
            auto* instancer = dynamic_cast<HdNoorRayInstancer*>(
                delegate->GetRenderIndex().GetInstancer(GetInstancerId()));
            if (instancer != nullptr)
                transforms = instancer->ComputeInstanceTransforms(GetId());
            const GfMatrix4d prototypeTransform = delegate->GetTransform(GetId());
            for (GfMatrix4d& transform : transforms)
                transform = prototypeTransform * transform;
        }
        while (objectIds_.size() > transforms.size()) {
            scene.removeObject(objectIds_.back());
            objectIds_.pop_back();
        }
        while (objectIds_.size() < transforms.size()) {
            auto instance = std::make_unique<MeshInstance>(
                scene, GetId().GetString(), meshIndex_, Transform());
            objectIds_.push_back(scene.add(std::move(instance)));
        }
        for (size_t i = 0; i < transforms.size(); ++i)
            if (SceneObject* object = scene.getObject(objectIds_[i]))
                object->setWorldTransformFromMatrix(ToGlm(transforms[i]));
    }

    if (*dirtyBits & HdChangeTracker::DirtyVisibility) {
        const bool visible = delegate->GetVisible(GetId());
        for (const uint64_t objectId : objectIds_)
            if (SceneObject* object = scene.getObject(objectId))
                object->setVisible(visible);
        scene.setDirtyFlag(TLAS);
        scene.setDirtyFlag(Accumulation);
    }
    UpdateRenderTag(delegate, renderParam);
    param.MarkSceneDirty();
    *dirtyBits = HdChangeTracker::Clean;
}

void HdNoorRayMesh::Finalize(HdRenderParam* renderParam)
{
    auto& param = *static_cast<HdNoorRayRenderParam*>(renderParam);
    std::scoped_lock lock(param.mutex);
    if (meshIndex_ != ~0u)
        param.UnbindMaterial(boundMaterialId_, meshIndex_);
    boundMaterialId_ = SdfPath();
    for (const uint64_t objectId : objectIds_)
        param.session.scene.removeObject(objectId);
    objectIds_.clear();
    param.MarkSceneDirty();
}

PXR_NAMESPACE_CLOSE_SCOPE
