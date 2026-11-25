#pragma once

#include <array>
#include <optional>
#include <vector>

#include <RmlUi/Core/Mesh.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Controllers/Selection.h"
#include "NoorRay/Core/Math.h"
#include "NoorRay/Scene/Camera.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::App {

//! The viewport camera in the pixels of the viewport content box.
class OverlayView {
public:
    OverlayView(const Scene::Camera& camera, Rml::Vector2f size);

    struct Projected {
        Rml::Vector2f position;
        //! Viewport pixels per world unit at the point.
        float pixelsPerUnit;
        float depth;
    };
    std::optional<Projected> project(Core::float3 world) const;

    struct Ray {
        Core::float3 origin;
        Core::float3 direction;
    };
    Ray rayAt(Rml::Vector2f pixel) const;
    Core::float3 getForward() const { return forward; }

private:
    Core::Transform cameraToWorld;
    Core::Transform worldToCamera;
    Core::float3 forward;
    bool orthographic;
    float halfHeight;
    float aspect;
    float near;
    Rml::Vector2f size;
};

//! The transform the gizmo edits: translate along world axes, scale along the object's own axes, or rotate about world axes.
enum class GizmoMode { Translate, Scale, Rotate };

//! The gizmo part under the mouse: an axis, the plane normal to an axis, the centre (moves in the view plane or scales evenly), or the ring about an axis.
struct GizmoPart {
    enum class Kind { Axis, Plane, Center, Ring };
    Kind kind;
    int axis = 0;
    bool operator==(const GizmoPart&) const = default;
};

//! After ImGuizmo's gizmos and the light billboards of the old Vulkan viewport, tessellated into RmlUi geometry over the rendered image.
class ViewportOverlay {
public:
    static constexpr int iconCount = 5;
    static int iconCell(float density);
    //! The icons of the lights and the camera in one row of cells, as premultiplied RGBA: the neutral body, then a row with the glow.
    static std::vector<Rml::byte> paintIconAtlas(int cell);
    //! The textured quads of the light and camera icons, the farthest first, tinted by the colour of the light.
    Rml::Mesh buildIcons(const Scene::Scene& scene, const OverlayView& view, float density) const;
    //! Builds the gizmo on the selection.
    Rml::Mesh build(const Scene::Scene& scene, const OverlayView& view, const Selection& selection, std::optional<GizmoPart> highlighted, float density,
                    GizmoMode mode);

    std::optional<GizmoPart> hitGizmo(const OverlayView& view, const Core::Transform& target, Rml::Vector2f mouse, float density, GizmoMode mode) const;
    //! The light or camera whose icon is under the mouse, the nearest first.
    std::optional<Selection> hitIcon(const Scene::Scene& scene, const OverlayView& view, Rml::Vector2f mouse, float density) const;

    //! Starts a drag of the part; the gizmo keeps the transform it had when the drag began.
    bool beginDrag(const OverlayView& view, GizmoPart part, const Core::Transform& target, Rml::Vector2f mouse, float density, GizmoMode mode);
    //! The new transform for the mouse, or nothing if the ray misses the drag plane.
    std::optional<Core::Transform> drag(const OverlayView& view, Rml::Vector2f mouse) const;
    void endDrag() { dragged.reset(); }
    std::optional<GizmoPart> getDragged() const { return dragged; }

private:
    struct Gizmo {
        Rml::Vector2f center;
        std::array<Rml::Vector2f, 3> ends;
        std::array<bool, 3> axisVisible;
        std::array<std::array<Rml::Vector2f, 4>, 3> planes;
        std::array<bool, 3> planeVisible;
        //! The rings of the rotate gizmo as closed polylines; a point is missing where it is behind the camera.
        std::array<std::vector<std::optional<Rml::Vector2f>>, 3> rings;
        std::array<std::vector<bool>, 3> ringFront;
        //! Pixels per dp of the gizmo at the depth of its origin.
        float scale;
        float length;
    };
    std::optional<Gizmo> layout(const OverlayView& view, const Core::Transform& target, float density, GizmoMode mode) const;
    std::optional<Core::float3> pointOnDrag(const OverlayView& view, Rml::Vector2f mouse) const;

    std::optional<GizmoPart> dragged;
    GizmoMode dragMode = GizmoMode::Translate;
    Core::Transform dragTransform;
    std::array<Core::float3, 3> dragAxes{};
    Core::float3 dragOrigin{};
    Core::float3 dragStart{};
    Core::float3 dragNormal{};
    Rml::Vector2f dragMouse;
    float dragDensity = 1.0f;
};

//! The world transform of a light, a camera other than the active one, an instance or a Gaussian cloud.
std::optional<Core::Transform> transformOf(const Scene::Scene& scene, const Selection& selection);
void setTransform(Scene::Scene& scene, const Selection& selection, const Core::Transform& transform);
//! Places a node in world space below its parent; its children and object follow.
void setNodeWorldTransform(Scene::Scene& scene, Scene::NodeHandle handle, const Core::Transform& world);

}
