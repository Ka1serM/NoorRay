#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <glm/vec3.hpp>

// The icons the viewport draws. The ids index the icon textures; the first
// four are LightInstance's light types.
enum class BillboardIcon : uint32_t {
    PointLight,
    SpotLight,
    RectLight,
    DirectionalLight,
    SkyLight,
    Camera,
    Audio,
    PlayerStart,
    Fog,
    Atmosphere,
    Cloud,
    ReflectionCapture,
    Decal,
    Note,
    TargetPoint,
    Particles,
    Wind,
    SceneCapture,
    Constraint,
    RadialForce,
    Marker,
};

// The texture of each icon, by icon id: assets/billboards/<name>.msdf.
inline constexpr auto BillboardIconNames = std::to_array<std::string_view>({
    "point_light", "spot_light", "rect_light", "directional_light", "sky_light",
    "camera", "audio", "player_start", "fog", "atmosphere", "cloud",
    "reflection_capture", "decal", "note", "target_point", "particles", "wind",
    "scene_capture", "constraint", "radial_force", "marker"});
static_assert(BillboardIconNames.size() == static_cast<std::size_t>(BillboardIcon::Marker) + 1);

// The camera-facing icon the viewport draws at a scene object's world
// position. Any scene object may carry one.
struct Billboard {
    BillboardIcon icon{BillboardIcon::Marker};
    glm::vec3 color{1.0f};
};
