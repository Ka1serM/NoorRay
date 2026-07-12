#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace nr::sceneio {

using Vec3 = std::array<float, 3>;
using Resolution = std::array<uint32_t, 2>;

struct EnvironmentFile {
    Vec3 color{1.0f, 1.0f, 1.0f};
    float lighting_exposure{1.0f};
    float visible_exposure{};
    bool visible{true};
};

struct RenderSettingsFile {
    int max_samples{3000};
    int gaussian_shading_mode{};
    // Accepted for compatibility with older scene files; no longer applied or written.
    std::optional<int> gaussian_import_sh_degree;
    int gaussian_render_sh_degree{3};
};

struct CameraFile {
    std::string type{"perspective"};
    Vec3 position{};
    Vec3 rotation_euler{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    float focal_length{50.0f};
    float focus_distance{2.0f};
    float bokeh_bias{1.0f};
    float aperture_diameter{};
    Resolution resolution{1280, 720};
    std::string lens;
    std::string sensor;
    std::string glass_catalogs;
};

struct MaterialFile {
    Vec3 albedo{0.8f, 0.8f, 0.8f};
    float roughness{0.5f};
    float metallic{};
    float specular{1.0f};
    float ior{1.5f};
    float ior_r{1.5f};
    float ior_g{1.5f};
    float ior_b{1.5f};
    float transmission{};
    float opacity{1.0f};
    Vec3 emission{};
    float emission_strength{};
};

struct ObjectFile {
    std::string type{"sphere"};
    std::string name{"Object"};
    std::string path;
    Vec3 position{};
    Vec3 rotation_euler{};
    Vec3 scale{1.0f, 1.0f, 1.0f};
    std::optional<MaterialFile> material;
};

struct SceneFile {
    std::optional<EnvironmentFile> environment;
    std::optional<RenderSettingsFile> render_settings;
    std::optional<CameraFile> camera;
    std::vector<ObjectFile> objects;
};

} // namespace nr::sceneio
