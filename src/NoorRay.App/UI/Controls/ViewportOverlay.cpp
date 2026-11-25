#include "NoorRay.App/UI/Controls/ViewportOverlay.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>

#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <lunasvg.h>

namespace NoorRay::App {

namespace {

using Vec2 = Rml::Vector2f;

//! The axis colours of ImGuizmo in the old viewport, and its yellow selection.
const Rml::Colourb axisColors[3] = {{233, 62, 85}, {140, 206, 40}, {49, 155, 249}};
const Rml::Colourb highlightColor{255, 255, 0};
const Rml::Colourb shadowColor{0, 0, 0, 140};

constexpr float gizmoLength = 96.0f;
constexpr float lineWidth = 3.0f;
constexpr float arrowLength = 15.0f;
constexpr float arrowHalfWidth = 6.5f;
constexpr float centerRadius = 6.0f;
constexpr float planeNear = 0.3f;
constexpr float planeFar = 0.55f;
constexpr float hitSlop = 7.0f;

//! Icon radius in dp: the largest at iconNear and closer, the smallest at iconFar and beyond.
constexpr float iconMinRadius = 18.0f;
constexpr float iconMaxRadius = 32.0f;
constexpr float iconNear = 2.0f;
constexpr float iconFar = 60.0f;
constexpr float iconPick = 0.8f;

Rml::ColourbPremultiplied premultiplied(const Rml::Colourb colour, const float alpha = 1.0f) {
    return Rml::Colourb(colour.red, colour.green, colour.blue, static_cast<Rml::byte>(colour.alpha * alpha)).ToPremultiplied();
}

float cross2(const Vec2 a, const Vec2 b) {
    return a.x * b.y - a.y * b.x;
}

float segmentDistance(const Vec2 p, const Vec2 a, const Vec2 b) {
    const Vec2 ab = b - a;
    const float lengthSquared = ab.DotProduct(ab);
    const float t = lengthSquared > 0.0f ? std::clamp((p - a).DotProduct(ab) / lengthSquared, 0.0f, 1.0f) : 0.0f;
    return (p - (a + ab * t)).Magnitude();
}

bool insideConvex(const Vec2 p, const std::array<Vec2, 4>& corners) {
    float sign = 0.0f;
    for (std::size_t i = 0; i < corners.size(); ++i) {
        const float side = cross2(corners[(i + 1) % corners.size()] - corners[i], p - corners[i]);
        if (side == 0.0f)
            continue;
        if (sign != 0.0f && (side > 0.0f) != (sign > 0.0f))
            return false;
        sign = side;
    }
    return true;
}

class Painter {
public:
    explicit Painter(Rml::Mesh& mesh) : mesh(mesh) {}

    void polygon(std::initializer_list<Vec2> points, const Rml::ColourbPremultiplied colour) {
        const int first = static_cast<int>(mesh.vertices.size());
        for (const Vec2 point : points)
            mesh.vertices.push_back({point, colour, {0.0f, 0.0f}});
        for (int i = 1; i + 1 < static_cast<int>(points.size()); ++i)
            mesh.indices.insert(mesh.indices.end(), {first, first + i, first + i + 1});
    }

    void line(const Vec2 a, const Vec2 b, const float width, const Rml::ColourbPremultiplied colour) {
        const Vec2 along = b - a;
        const float length = along.Magnitude();
        if (length <= 0.0f)
            return;
        const Vec2 side = Vec2(-along.y, along.x) * (0.5f * width / length);
        polygon({a + side, b + side, b - side, a - side}, colour);
    }

    void disc(const Vec2 center, const float radius, const Rml::ColourbPremultiplied colour) {
        const int segments = std::clamp(static_cast<int>(radius * 2.0f), 10, 32);
        const int first = static_cast<int>(mesh.vertices.size());
        mesh.vertices.push_back({center, colour, {0.0f, 0.0f}});
        for (int i = 0; i < segments; ++i) {
            const float angle = 2.0f * Core::pi * float(i) / float(segments);
            mesh.vertices.push_back({center + Vec2(std::cos(angle), std::sin(angle)) * radius, colour, {0.0f, 0.0f}});
        }
        for (int i = 0; i < segments; ++i)
            mesh.indices.insert(mesh.indices.end(), {first, first + 1 + i, first + 1 + (i + 1) % segments});
    }

private:
    Rml::Mesh& mesh;
};

constexpr std::array<const char*, ViewportOverlay::iconCount> iconSources{"/UI/Icons/light-point.svg", "/UI/Icons/light-spot.svg", "/UI/Icons/light-sun.svg",
                                                                          "/UI/Icons/light-area.svg", "/UI/Icons/object-camera.svg"};

int lightIcon(const Scene::LightType type) {
    switch (type) {
    case Scene::LightType::Spot:
        return 1;
    case Scene::LightType::Directional:
        return 2;
    case Scene::LightType::Rect:
    case Scene::LightType::Disk:
        return 3;
    case Scene::LightType::Point:
    case Scene::LightType::Sphere:
        break;
    }
    return 0;
}

constexpr int cameraIcon = 4;

//! layer 0 is the neutral body of the icon, drawn as it is; layer 1 is its glow as brightness alone, which the colour tints.
void paintIconLayer(Rml::Mesh& mesh, const int icon, const int layer, const Vec2 center, const float radius, const Rml::ColourbPremultiplied colour) {
    const int first = static_cast<int>(mesh.vertices.size());
    const float left = static_cast<float>(icon) / ViewportOverlay::iconCount;
    const float right = static_cast<float>(icon + 1) / ViewportOverlay::iconCount;
    const float top = 0.5f * static_cast<float>(layer);
    const float bottom = top + 0.5f;
    mesh.vertices.push_back({center + Vec2(-radius, -radius), colour, {left, top}});
    mesh.vertices.push_back({center + Vec2(radius, -radius), colour, {right, top}});
    mesh.vertices.push_back({center + Vec2(radius, radius), colour, {right, bottom}});
    mesh.vertices.push_back({center + Vec2(-radius, radius), colour, {left, bottom}});
    mesh.indices.insert(mesh.indices.end(), {first, first + 1, first + 2, first, first + 2, first + 3});
}

void paintIcon(Rml::Mesh& mesh, const int icon, const Vec2 center, const float radius, const Rml::ColourbPremultiplied colour) {
    paintIconLayer(mesh, icon, 0, center, radius, Rml::Colourb(255, 255, 255).ToPremultiplied());
    paintIconLayer(mesh, icon, 1, center, radius, colour);
}

Rml::Colourb lightColour(const Core::float3 color) {
    const float scale = 1.0f / std::max(Core::maxComponent(color), 1e-4f);
    const auto channel = [&](const float value) { return static_cast<Rml::byte>(std::lround(255.0f * std::clamp(value * scale, 0.0f, 1.0f))); };
    return {channel(color.x), channel(color.y), channel(color.z), 255};
}

//! Overlay pixels per dp at a depth: full size up close, down to the ratio of the icon radii far away, so the icons and the gizmo shrink together.
float distanceScale(const float depth, const float density) {
    const float t = std::clamp((depth - iconNear) / (iconFar - iconNear), 0.0f, 1.0f);
    return (1.0f + (iconMinRadius / iconMaxRadius - 1.0f) * t) * density;
}

float iconRadius(const float depth, const float density) {
    return iconMaxRadius * distanceScale(depth, density);
}

Core::float3 axisVector(const int axis) {
    return {axis == 0 ? 1.0f : 0.0f, axis == 1 ? 1.0f : 0.0f, axis == 2 ? 1.0f : 0.0f};
}

//! Scale uses the object axes, each turned toward its world axis, so it never shears and shows the handles of translate; translate and rotate use the world
//! axes.
std::array<Core::float3, 3> gizmoAxes(const Core::Transform& target, const GizmoMode mode) {
    std::array<Core::float3, 3> axes{axisVector(0), axisVector(1), axisVector(2)};
    if (mode != GizmoMode::Scale)
        return axes;
    for (int axis = 0; axis < 3; ++axis)
        if (const Core::float3 column = target.getColumn(axis); Core::length(column) > 1e-6f)
            axes[axis] = Core::normalize(column) * (Core::dot(column, axisVector(axis)) < 0.0f ? -1.0f : 1.0f);
    return axes;
}

Core::Transform rotation(const Core::float3 axis, const float angle) {
    const float half = 0.5f * angle;
    const Core::float3 imaginary = axis * std::sin(half);
    return Core::Transform::fromTrs({0.0f, 0.0f, 0.0f}, {imaginary.x, imaginary.y, imaginary.z, std::cos(half)}, {1.0f, 1.0f, 1.0f});
}

Core::Transform withTranslation(Core::Transform transform, const Core::float3 translation) {
    transform.rows[0][3] = translation.x;
    transform.rows[1][3] = translation.y;
    transform.rows[2][3] = translation.z;
    return transform;
}

Core::float3 withComponent(Core::float3 vector, const int index, const float value) {
    if (index == 0)
        vector.x = value;
    else if (index == 1)
        vector.y = value;
    else
        vector.z = value;
    return vector;
}

constexpr int ringSegments = 64;
constexpr float scaleBox = 5.0f;
constexpr float minimumScale = 1e-3f;
//! Pixels of vertical mouse travel that double or halve the size in an even scale.
constexpr float uniformScalePixels = 120.0f;

template <class Visit>
void forEachIcon(const Scene::Scene& scene, Visit&& visit) {
    const Core::SlotMap<Scene::Light>& lights = scene.getLights();
    for (std::uint32_t i = 0; i < lights.size(); ++i)
        visit(Selection{lights.keyAt(i)}, lights.getValues()[i].transform.getTranslation(), lightIcon(lights.getValues()[i].type),
              lightColour(lights.getValues()[i].color));
    const Core::SlotMap<Scene::Camera>& cameras = scene.getCameras();
    for (std::uint32_t i = 0; i < cameras.size(); ++i)
        if (cameras.keyAt(i) != scene.getActiveCamera())
            visit(Selection{cameras.keyAt(i)}, cameras.getValues()[i].transform.getTranslation(), cameraIcon, Rml::Colourb(255, 255, 255));
}

}

OverlayView::OverlayView(const Scene::Camera& camera, const Rml::Vector2f size)
    : cameraToWorld(camera.transform), worldToCamera(camera.transform.inverse()), forward(Core::normalize(-camera.transform.getColumn(2))),
      orthographic(camera.type == Scene::CameraType::Orthographic),
      halfHeight(orthographic ? 0.5f * camera.orthographicHeight : std::tan(0.5f * camera.verticalFov)), aspect(size.y > 0.0f ? size.x / size.y : 1.0f),
      near(std::max(camera.near, 1e-4f)), size(size) {}

std::optional<OverlayView::Projected> OverlayView::project(const Core::float3 world) const {
    const Core::float3 local = worldToCamera.transformPoint(world);
    const float depth = -local.z;
    if (!orthographic && depth <= near)
        return std::nullopt;
    const float extent = orthographic ? halfHeight : halfHeight * depth;
    const float x = local.x / (extent * aspect);
    const float y = local.y / extent;
    return Projected{{(x * 0.5f + 0.5f) * size.x, (0.5f - y * 0.5f) * size.y}, 0.5f * size.y / extent, depth};
}

OverlayView::Ray OverlayView::rayAt(const Rml::Vector2f pixel) const {
    const float x = (2.0f * pixel.x / std::max(size.x, 1.0f) - 1.0f) * halfHeight * aspect;
    const float y = (1.0f - 2.0f * pixel.y / std::max(size.y, 1.0f)) * halfHeight;
    if (orthographic)
        return {cameraToWorld.transformPoint({x, y, 0.0f}), forward};
    return {cameraToWorld.getTranslation(), Core::normalize(cameraToWorld.transformVector({x, y, -1.0f}))};
}

std::optional<ViewportOverlay::Gizmo> ViewportOverlay::layout(const OverlayView& view, const Core::Transform& target, const float density,
                                                              const GizmoMode mode) const {
    const Core::float3 origin = target.getTranslation();
    const std::optional<OverlayView::Projected> center = view.project(origin);
    if (!center)
        return std::nullopt;
    const std::array<Core::float3, 3> axes = gizmoAxes(target, mode);
    Gizmo gizmo{};
    gizmo.center = center->position;
    gizmo.scale = distanceScale(center->depth, density);
    gizmo.length = gizmoLength * gizmo.scale;
    const float worldLength = gizmo.length / center->pixelsPerUnit;
    if (mode == GizmoMode::Rotate) {
        for (int normal = 0; normal < 3; ++normal) {
            const Core::float3 u = axes[(normal + 1) % 3];
            const Core::float3 v = axes[(normal + 2) % 3];
            for (int i = 0; i < ringSegments; ++i) {
                const float angle = 2.0f * Core::pi * float(i) / float(ringSegments);
                const Core::float3 offset = (u * std::cos(angle) + v * std::sin(angle)) * worldLength;
                const std::optional<OverlayView::Projected> point = view.project(origin + offset);
                gizmo.rings[normal].push_back(point ? std::optional(point->position) : std::nullopt);
                gizmo.ringFront[normal].push_back(Core::dot(offset, view.getForward()) <= 0.0f);
            }
        }
        return gizmo;
    }
    std::array<std::optional<Vec2>, 3> ends;
    for (int axis = 0; axis < 3; ++axis) {
        const std::optional<OverlayView::Projected> end = view.project(origin + axes[axis] * worldLength);
        ends[axis] = end ? std::optional(end->position) : std::nullopt;
        gizmo.axisVisible[axis] = end && (end->position - gizmo.center).Magnitude() > 0.15f * gizmo.length;
        gizmo.ends[axis] = end ? end->position : gizmo.center;
    }
    for (int normal = 0; normal < 3; ++normal) {
        const int u = (normal + 1) % 3;
        const int v = (normal + 2) % 3;
        gizmo.planeVisible[normal] = std::abs(Core::dot(axes[normal], view.getForward())) > 0.2f && ends[u] && ends[v];
        if (!gizmo.planeVisible[normal])
            continue;
        const auto corner = [&](const float a, const float b) {
            const std::optional<OverlayView::Projected> point = view.project(origin + (axes[u] * a + axes[v] * b) * worldLength);
            return point ? point->position : gizmo.center;
        };
        gizmo.planes[normal] = {corner(planeNear, planeNear), corner(planeFar, planeNear), corner(planeFar, planeFar), corner(planeNear, planeFar)};
    }
    return gizmo;
}

Rml::Mesh ViewportOverlay::build(const Scene::Scene& scene, const OverlayView& view, const Selection& selection, const std::optional<GizmoPart> highlighted,
                                 const float density, const GizmoMode mode) {
    Rml::Mesh mesh;
    Painter painter(mesh);
    const std::optional<Core::Transform> target = transformOf(scene, selection);
    if (!target)
        return mesh;
    const std::optional<Gizmo> gizmo = layout(view, *target, density, mode);
    if (!gizmo)
        return mesh;
    const std::optional<GizmoPart> active = dragged ? dragged : highlighted;
    const auto colourOf = [&](const GizmoPart part) { return active == part ? highlightColor : axisColors[part.axis]; };
    if (mode == GizmoMode::Rotate) {
        //! The back half of each ring is drawn first and faint, so the front halves cross over it.
        for (const bool front : {false, true})
            for (int normal = 0; normal < 3; ++normal) {
                const auto& ring = gizmo->rings[normal];
                const Rml::Colourb colour = colourOf({GizmoPart::Kind::Ring, normal});
                for (int i = 0; i < ringSegments; ++i) {
                    const int next = (i + 1) % ringSegments;
                    if (!ring[i] || !ring[next] || gizmo->ringFront[normal][i] != front)
                        continue;
                    painter.line(*ring[i], *ring[next], (front ? lineWidth : 1.5f) * gizmo->scale, premultiplied(colour, front ? 1.0f : 0.35f));
                }
            }
        painter.disc(gizmo->center, 3.0f * gizmo->scale, premultiplied(Rml::Colourb(255, 255, 255, 200)));
        return mesh;
    }
    for (int normal = 0; normal < 3; ++normal) {
        if (!gizmo->planeVisible[normal])
            continue;
        const auto& corners = gizmo->planes[normal];
        const Rml::Colourb colour = colourOf({GizmoPart::Kind::Plane, normal});
        painter.polygon({corners[0], corners[1], corners[2], corners[3]}, premultiplied(colour, 0.45f));
        for (std::size_t i = 0; i < corners.size(); ++i)
            painter.line(corners[i], corners[(i + 1) % corners.size()], 1.0f * gizmo->scale, premultiplied(colour));
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (!gizmo->axisVisible[axis])
            continue;
        const Rml::ColourbPremultiplied colour = premultiplied(colourOf({GizmoPart::Kind::Axis, axis}));
        const Vec2 along = gizmo->ends[axis] - gizmo->center;
        const Vec2 direction = along / along.Magnitude();
        const Vec2 side(-direction.y, direction.x);
        if (mode == GizmoMode::Scale) {
            const float half = scaleBox * gizmo->scale;
            const Vec2 end = gizmo->ends[axis];
            painter.line(gizmo->center, end - direction * half, lineWidth * gizmo->scale, colour);
            painter.polygon(
                {end + (direction + side) * half, end + (direction - side) * half, end - (direction + side) * half, end - (direction - side) * half}, colour);
            continue;
        }
        const Vec2 base = gizmo->ends[axis] - direction * (arrowLength * gizmo->scale);
        painter.line(gizmo->center, base, lineWidth * gizmo->scale, colour);
        painter.polygon({gizmo->ends[axis], base + side * (arrowHalfWidth * gizmo->scale), base - side * (arrowHalfWidth * gizmo->scale)}, colour);
    }
    const bool centerActive = active == GizmoPart{GizmoPart::Kind::Center, 0};
    painter.disc(gizmo->center, (centerRadius + 1.5f) * gizmo->scale, premultiplied(shadowColor));
    painter.disc(gizmo->center, centerRadius * gizmo->scale, premultiplied(centerActive ? highlightColor : Rml::Colourb(255, 255, 255, 230)));
    return mesh;
}

std::optional<GizmoPart> ViewportOverlay::hitGizmo(const OverlayView& view, const Core::Transform& target, const Vec2 mouse, const float density,
                                                   const GizmoMode mode) const {
    const std::optional<Gizmo> gizmo = layout(view, target, density, mode);
    if (!gizmo)
        return std::nullopt;
    const float slop = hitSlop * gizmo->scale;
    if (mode == GizmoMode::Rotate) {
        std::optional<GizmoPart> nearest;
        float best = slop;
        for (const bool front : {true, false})
            for (int normal = 0; normal < 3; ++normal)
                for (int i = 0; i < ringSegments; ++i) {
                    const auto& ring = gizmo->rings[normal];
                    const int next = (i + 1) % ringSegments;
                    if (!ring[i] || !ring[next] || gizmo->ringFront[normal][i] != front)
                        continue;
                    if (const float distance = segmentDistance(mouse, *ring[i], *ring[next]); distance <= best) {
                        best = distance;
                        nearest = GizmoPart{GizmoPart::Kind::Ring, normal};
                    }
                }
        return nearest;
    }
    if ((mouse - gizmo->center).Magnitude() <= (centerRadius + 3.0f) * gizmo->scale)
        return GizmoPart{GizmoPart::Kind::Center, 0};
    for (int axis = 0; axis < 3; ++axis)
        if (gizmo->axisVisible[axis] && segmentDistance(mouse, gizmo->center, gizmo->ends[axis]) <= slop)
            return GizmoPart{GizmoPart::Kind::Axis, axis};
    for (int normal = 0; normal < 3; ++normal)
        if (gizmo->planeVisible[normal] && insideConvex(mouse, gizmo->planes[normal]))
            return GizmoPart{GizmoPart::Kind::Plane, normal};
    return std::nullopt;
}

Rml::Mesh ViewportOverlay::buildIcons(const Scene::Scene& scene, const OverlayView& view, const float density) const {
    struct Placed {
        int icon;
        Vec2 center;
        float radius;
        float depth;
        Rml::ColourbPremultiplied colour;
    };
    std::vector<Placed> placed;
    forEachIcon(scene, [&](const Selection&, const Core::float3 position, const int icon, const Rml::Colourb colour) {
        if (const std::optional<OverlayView::Projected> projected = view.project(position))
            placed.push_back({icon, projected->position, iconRadius(projected->depth, density), projected->depth, premultiplied(colour)});
    });
    std::ranges::sort(placed, std::greater{}, &Placed::depth);
    Rml::Mesh mesh;
    mesh.vertices.reserve(placed.size() * 4);
    mesh.indices.reserve(placed.size() * 6);
    for (const Placed& icon : placed)
        paintIcon(mesh, icon.icon, icon.center, icon.radius, icon.colour);
    return mesh;
}

int ViewportOverlay::iconCell(const float density) {
    return static_cast<int>(std::ceil(2.0f * iconMaxRadius * density));
}

std::vector<Rml::byte> ViewportOverlay::paintIconAtlas(const int cell) {
    std::vector<Rml::byte> pixels(static_cast<std::size_t>(cell) * cell * iconCount * 4 * 2);
    const std::size_t glowOffset = static_cast<std::size_t>(cell) * cell * iconCount * 4;
    const int rowPixels = cell * iconCount;
    //! A one pixel gutter keeps the bilinear filter from reading the neighbouring cell.
    const int inner = cell - 2;
    for (int icon = 0; icon < iconCount; ++icon) {
        Rml::String file;
        if (!Rml::GetFileInterface()->LoadFile(iconSources[icon], file)) {
            Rml::Log::Message(Rml::Log::LT_ERROR, "The overlay icon '%s' could not be read.", iconSources[icon]);
            continue;
        }
        const std::unique_ptr<lunasvg::Document> document = lunasvg::Document::loadFromData(file.data(), file.size());
        if (!document) {
            Rml::Log::Message(Rml::Log::LT_ERROR, "The overlay icon '%s' is not a valid SVG file.", iconSources[icon]);
            continue;
        }
        const float scale = static_cast<float>(inner) / std::max(document->width(), document->height());
        const int width = std::max(1, static_cast<int>(std::lround(document->width() * scale)));
        const int height = std::max(1, static_cast<int>(std::lround(document->height() * scale)));
        const lunasvg::Bitmap bitmap = document->renderToBitmap(width, height);
        const int left = icon * cell + (cell - width) / 2;
        const int top = (cell - height) / 2;
        for (int y = 0; y < height; ++y) {
            const auto* const row = reinterpret_cast<const std::uint32_t*>(bitmap.data() + static_cast<std::ptrdiff_t>(y) * bitmap.stride());
            Rml::byte* body = pixels.data() + (static_cast<std::size_t>(top + y) * rowPixels + left) * 4;
            Rml::byte* glow = body + glowOffset;
            for (int x = 0; x < width; ++x) {
                //! Saturated pixels are the glow of the icon: they keep their brightness and lose their hue, so the light colour alone tints them.
                const std::uint32_t argb = row[x];
                const float red = static_cast<float>((argb >> 16) & 0xffu);
                const float green = static_cast<float>((argb >> 8) & 0xffu);
                const float blue = static_cast<float>(argb & 0xffu);
                const float alpha = static_cast<float>(argb >> 24);
                const float high = std::max({red, green, blue});
                const float saturation = high > 0.0f ? (high - std::min({red, green, blue})) / high : 0.0f;
                const float share = std::clamp((saturation - 0.12f) / 0.18f, 0.0f, 1.0f);
                const auto byte = [](const float value) { return static_cast<Rml::byte>(std::lround(std::clamp(value, 0.0f, 255.0f))); };
                for (const float channel : {red, green, blue})
                    *body++ = byte(channel * (1.0f - share));
                *body++ = byte(alpha * (1.0f - share));
                for (int channel = 0; channel < 3; ++channel)
                    *glow++ = byte(high * share);
                *glow++ = byte(alpha * share);
            }
        }
    }
    return pixels;
}

std::optional<Selection> ViewportOverlay::hitIcon(const Scene::Scene& scene, const OverlayView& view, const Vec2 mouse, const float density) const {
    std::optional<Selection> hit;
    float nearest = 0.0f;
    forEachIcon(scene, [&](const Selection& owner, const Core::float3 position, int, Rml::Colourb) {
        const std::optional<OverlayView::Projected> projected = view.project(position);
        if (!projected || (mouse - projected->position).Magnitude() > iconPick * iconRadius(projected->depth, density))
            return;
        if (hit && projected->depth >= nearest)
            return;
        hit = owner;
        nearest = projected->depth;
    });
    return hit;
}

bool ViewportOverlay::beginDrag(const OverlayView& view, const GizmoPart part, const Core::Transform& target, const Vec2 mouse, const float density,
                                const GizmoMode mode) {
    dragged = part;
    dragMode = mode;
    dragTransform = target;
    dragAxes = gizmoAxes(target, mode);
    dragOrigin = target.getTranslation();
    dragMouse = mouse;
    dragDensity = density;
    const Core::float3 forward = view.getForward();
    if (part.kind == GizmoPart::Kind::Center)
        dragNormal = forward;
    else if (part.kind == GizmoPart::Kind::Plane || part.kind == GizmoPart::Kind::Ring)
        dragNormal = dragAxes[part.axis];
    else {
        //! The axis moves in the plane through it that faces the camera most.
        const Core::float3 axis = dragAxes[part.axis];
        const Core::float3 across = Core::cross(axis, forward);
        dragNormal = Core::length(across) > 1e-4f ? Core::normalize(Core::cross(across, axis)) : forward;
    }
    const std::optional<Core::float3> start = pointOnDrag(view, mouse);
    if (!start) {
        dragged.reset();
        return false;
    }
    dragStart = *start;
    return true;
}

std::optional<Core::float3> ViewportOverlay::pointOnDrag(const OverlayView& view, const Vec2 mouse) const {
    const OverlayView::Ray ray = view.rayAt(mouse);
    const float facing = Core::dot(ray.direction, dragNormal);
    if (std::abs(facing) < 1e-5f)
        return std::nullopt;
    const float distance = Core::dot(dragOrigin - ray.origin, dragNormal) / facing;
    if (distance < 0.0f)
        return std::nullopt;
    return ray.origin + ray.direction * distance;
}

std::optional<Core::Transform> ViewportOverlay::drag(const OverlayView& view, const Vec2 mouse) const {
    if (!dragged)
        return std::nullopt;
    if (dragMode == GizmoMode::Scale && dragged->kind == GizmoPart::Kind::Center) {
        const float factor = std::exp((dragMouse.y - mouse.y) / (uniformScalePixels * dragDensity));
        return dragTransform * Core::Transform::fromTrs({0.0f, 0.0f, 0.0f}, {}, {factor, factor, factor});
    }
    const std::optional<Core::float3> point = pointOnDrag(view, mouse);
    if (!point)
        return std::nullopt;
    const Core::float3 from = dragStart - dragOrigin;
    const Core::float3 to = *point - dragOrigin;
    if (dragMode == GizmoMode::Rotate) {
        const Core::float3 axis = dragAxes[dragged->axis];
        const float angle = std::atan2(Core::dot(Core::cross(from, to), axis), Core::dot(from, to));
        return withTranslation(rotation(axis, angle) * dragTransform, dragOrigin);
    }
    if (dragMode == GizmoMode::Scale) {
        Core::float3 factors{1.0f, 1.0f, 1.0f};
        const auto ratio = [](const float before, const float after) { return std::abs(before) > 1e-6f ? std::max(after / before, minimumScale) : 1.0f; };
        if (dragged->kind == GizmoPart::Kind::Axis) {
            const Core::float3 axis = dragAxes[dragged->axis];
            factors = withComponent(factors, dragged->axis, ratio(Core::dot(from, axis), Core::dot(to, axis)));
        } else {
            const Core::float3 normal = dragAxes[dragged->axis];
            const float factor = ratio(Core::length(from - normal * Core::dot(from, normal)), Core::length(to - normal * Core::dot(to, normal)));
            factors = withComponent(withComponent(factors, (dragged->axis + 1) % 3, factor), (dragged->axis + 2) % 3, factor);
        }
        return dragTransform * Core::Transform::fromTrs({0.0f, 0.0f, 0.0f}, {}, factors);
    }
    Core::float3 moved = *point - dragStart;
    if (dragged->kind == GizmoPart::Kind::Axis) {
        const Core::float3 axis = dragAxes[dragged->axis];
        moved = axis * Core::dot(moved, axis);
    }
    return withTranslation(dragTransform, dragOrigin + moved);
}

std::optional<Core::Transform> transformOf(const Scene::Scene& scene, const Selection& selection) {
    if (const auto* handle = std::get_if<Scene::InstanceHandle>(&selection)) {
        const Scene::Instance* instance = scene.find(*handle);
        return instance ? std::optional(instance->transform) : std::nullopt;
    }
    if (const auto* handle = std::get_if<Scene::LightHandle>(&selection)) {
        const Scene::Light* light = scene.find(*handle);
        return light ? std::optional(light->transform) : std::nullopt;
    }
    if (const auto* handle = std::get_if<Scene::CameraHandle>(&selection)) {
        const Scene::Camera* camera = scene.find(*handle);
        return camera && *handle != scene.getActiveCamera() ? std::optional(camera->transform) : std::nullopt;
    }
    if (const auto* handle = std::get_if<Scene::GaussianCloudHandle>(&selection)) {
        const Scene::GaussianCloud* cloud = scene.find(*handle);
        return cloud ? std::optional(cloud->transform) : std::nullopt;
    }
    if (const auto* handle = std::get_if<Scene::NodeHandle>(&selection)) {
        const Scene::Node* node = scene.find(*handle);
        return node ? std::optional(node->worldTransform) : std::nullopt;
    }
    return std::nullopt;
}

void setTransform(Scene::Scene& scene, const Selection& selection, const Core::Transform& transform) {
    if (const auto* handle = std::get_if<Scene::InstanceHandle>(&selection)) {
        if (scene.find(*handle))
            scene.setInstanceTransform(*handle, transform);
    } else if (const auto* handle = std::get_if<Scene::LightHandle>(&selection)) {
        if (const Scene::Light* found = scene.find(*handle)) {
            Scene::Light light = *found;
            light.transform = transform;
            scene.setLight(*handle, light);
        }
    } else if (const auto* handle = std::get_if<Scene::CameraHandle>(&selection)) {
        if (const Scene::Camera* found = scene.find(*handle)) {
            Scene::Camera camera = *found;
            camera.previousTransform = camera.transform;
            camera.transform = transform;
            scene.setCamera(*handle, camera);
        }
    } else if (const auto* handle = std::get_if<Scene::GaussianCloudHandle>(&selection)) {
        if (Scene::GaussianCloud* cloud = scene.findForTraining(*handle)) {
            cloud->transform = transform;
            scene.markGaussiansChanged();
        }
    } else if (const auto* handle = std::get_if<Scene::NodeHandle>(&selection))
        setNodeWorldTransform(scene, *handle, transform);
}

void setNodeWorldTransform(Scene::Scene& scene, const Scene::NodeHandle handle, const Core::Transform& world) {
    const Scene::Node* node = scene.find(handle);
    if (!node)
        return;
    const Scene::Node* parent = scene.find(node->parent);
    scene.setLocalTransform(handle, parent ? parent->worldTransform.inverse() * world : world);
}

}
