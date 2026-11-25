#include "NoorRay.App/UI/Controls/ViewGizmo.h"

#include <algorithm>
#include <cmath>
#include <string>

#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Transform.h>
#include <RmlUi/Core/TransformPrimitive.h>

namespace NoorRay::App {

namespace {

//! These match the dp sizes in noorray.rcss.
constexpr float handleDistance = 51.0f;
constexpr float handleRadius = 12.0f;

const Rml::Element* findClass(const Rml::Element* element, const char* name) {
    for (; element; element = element->GetParentNode())
        if (element->IsClassSet(name))
            return element;
    return nullptr;
}

void setOrder(Rml::Element& element, const int from, const int to, const bool force) {
    if (!force && from == to)
        return;
    element.SetClass("order-" + std::to_string(from), false);
    element.SetClass("order-" + std::to_string(to), true);
}

}

bool ViewGizmo::find(Rml::Element& viewport) {
    if (found)
        return rotor != nullptr;
    found = true;
    valid = false;
    rotor = viewport.QuerySelector(".gizmo-rotor");
    if (!rotor)
        return false;
    const char* spokeNames[] = {".spoke-x", ".spoke-y", ".spoke-z"};
    for (int k = 0; k < 3; ++k)
        spokes[k] = rotor->QuerySelector(spokeNames[k]);
    Rml::ElementList handles;
    rotor->QuerySelectorAll(handles, ".gizmo-axis");
    axes.fill(nullptr);
    for (Rml::Element* const handle : handles)
        if (const int axis = handle->GetAttribute<int>("axis", -1); axis >= 0 && axis < 6)
            axes[axis] = handle;
    return true;
}

void ViewGizmo::sync(Rml::Element& viewport, const Core::Transform& camera) {
    if (!find(viewport))
        return;
    const Rml::Context* context = viewport.GetContext();
    const float density = context ? context->GetDensityIndependentPixelRatio() : 1.0f;
    const Core::float3 right = Core::normalize(camera.getColumn(0));
    const Core::float3 up = Core::normalize(camera.getColumn(1));
    const Core::float3 back = Core::normalize(camera.getColumn(2));
    const std::array<float, 9> next{right.x, right.y, right.z, up.x, up.y, up.z, back.x, back.y, back.z};
    if (valid && density == ratio && std::equal(next.begin(), next.end(), basis.begin(), [](const float a, const float b) { return std::abs(a - b) <= 1e-6f; }))
        return;
    const bool force = !valid;
    basis = next;
    ratio = density;
    valid = true;

    //! World to screen in RmlUi's frame (x right, y down, z toward the viewer); orthonormal, so the transpose inverts it.
    const Rml::Matrix4f rotation =
        Rml::Matrix4f::FromRows({right.x, right.y, right.z, 0.0f}, {-up.x, -up.y, -up.z, 0.0f}, {back.x, back.y, back.z, 0.0f}, {0, 0, 0, 1});
    const Rml::Matrix4f inverse =
        Rml::Matrix4f::FromColumns({right.x, right.y, right.z, 0.0f}, {-up.x, -up.y, -up.z, 0.0f}, {back.x, back.y, back.z, 0.0f}, {0, 0, 0, 1});
    rotor->SetProperty(Rml::PropertyId::Transform, Rml::Transform::MakeProperty({Rml::Transforms::Matrix3D(rotation)}));

    std::array<float, 6> depth{};
    for (int i = 0; i < 6; ++i)
        depth[i] = (i % 2 == 0 ? 1.0f : -1.0f) * next[6 + i / 2];
    std::array<int, 6> order{0, 1, 2, 3, 4, 5};
    std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) { return depth[a] < depth[b]; });
    std::array<int, 6> nextRank{};
    for (int r = 0; r < 6; ++r)
        nextRank[order[r]] = r;

    const float distance = handleDistance * density;
    const float radius = handleRadius * density;
    for (int i = 0; i < 6; ++i) {
        Rml::Element* const axis = axes[i];
        if (!axis)
            continue;
        const int k = i / 2;
        const float sign = i % 2 == 0 ? 1.0f : -1.0f;
        const Rml::Vector3f direction(k == 0 ? sign : 0.0f, k == 1 ? sign : 0.0f, k == 2 ? sign : 0.0f);
        //! Out along the axis in world space, then the rotor undone, so the handle faces the viewer upright.
        axis->SetProperty(Rml::PropertyId::Transform,
                          Rml::Transform::MakeProperty({Rml::Transforms::Translate3D(direction.x * distance, direction.y * distance, direction.z * distance),
                                                        Rml::Transforms::Matrix3D(inverse)}));
        setOrder(*axis, rank[i], nextRank[i], force);
        const bool turnedAway = depth[i] < -0.1f;
        if (force || turnedAway != behind[i])
            axis->SetClass("behind", turnedAway);
        behind[i] = turnedAway;
        if (sign < 0.0f || !spokes[k])
            continue;
        //! The spoke lies along its axis, turned about it to face the viewer, and stops one handle radius short on screen.
        const float screenLength = distance * std::sqrt(std::max(0.0f, 1.0f - depth[i] * depth[i]));
        const float lengthScale = screenLength > radius ? 1.0f - radius / screenLength : 0.0f;
        const Rml::Vector3f view(back.x, back.y, back.z);
        Rml::Vector3f across = view.CrossProduct(direction);
        const float acrossLength = across.Magnitude();
        across = acrossLength > 1e-4f ? across / acrossLength : Rml::Vector3f(right.x, right.y, right.z);
        const Rml::Vector3f normal = direction.CrossProduct(across);
        const Rml::Vector3f along = direction * lengthScale;
        spokes[k]->SetProperty(
            Rml::PropertyId::Transform,
            Rml::Transform::MakeProperty({Rml::Transforms::Matrix3D(Rml::Matrix4f::FromColumns(
                {along.x, along.y, along.z, 0.0f}, {across.x, across.y, across.z, 0.0f}, {normal.x, normal.y, normal.z, 0.0f}, {0, 0, 0, 1}))}));
        setOrder(*spokes[k], rank[i], nextRank[i], force);
    }
    rank = nextRank;
}

std::optional<ViewDrag> ViewGizmo::dragOf(const Rml::Element* element) {
    if (findClass(element, "tool-dolly"))
        return ViewDrag::Dolly;
    if (findClass(element, "tool-pan"))
        return ViewDrag::Pan;
    if (findClass(element, "gizmo-center") || findClass(element, "gizmo-axis"))
        return ViewDrag::Orbit;
    return std::nullopt;
}

int ViewGizmo::axisOf(const Rml::Element* element) {
    const Rml::Element* handle = findClass(element, "gizmo-axis");
    return handle ? handle->GetAttribute<int>("axis", -1) : -1;
}

}
