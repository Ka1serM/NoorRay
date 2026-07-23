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
};

struct RenderSettingsFile {
    int max_samples{3000};
    bool noise_limit_enabled{false};
    float noise_level{0.0001f};
    bool aov_enabled{true};
    bool optix_denoiser_enabled{false};
    int optix_denoiser_min_samples{1};
    int gaussian_shading_mode{};
    int gaussian_render_sh_degree{3};
};

struct CameraFile {
    std::string projection{"perspective"};
    bool active{};
    float focal_length_mm{50.0f};
    float focus_distance_cm{200.0f};
    float exposure{};
    float bokeh_bias{1.0f};
    float aperture_diameter_mm{};
    float sensor_width_mm{5.784f};
    float sensor_height_mm{3.264f};
    Resolution resolution{1280, 720};
    std::string lens;
    std::string sensor;
    std::string glass_catalogs;
    std::string sensor_type{"rectangular"};
    std::string psf;
    std::string ray_lut;
};

struct MaterialFile {
    Vec3 albedo{0.8f, 0.8f, 0.8f};
    float roughness{0.5f};
    float metallic{};
    float specular{1.0f};
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
    std::optional<CameraFile> camera;
    std::optional<MaterialFile> material;
};

struct SceneFile {
    std::optional<EnvironmentFile> environment;
    std::optional<RenderSettingsFile> render_settings;
    std::vector<ObjectFile> objects;
};

} // namespace nr::sceneio
