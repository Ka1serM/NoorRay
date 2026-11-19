#include "NoorRay.App/UI/Controllers/MaterialController.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <format>
#include <optional>
#include <string_view>
#include <variant>

#include "NoorRay.App/UI/Controls/FieldCommit.h"
#include "NoorRay.App/UI/Controls/ViewportElement.h"
#include "NoorRay.App/UI/Models/AppModel.h"
#include "RmlKit/UI/Backend/FileDialog.h"
#include "RmlKit/UI/Models/Model.h"

namespace NoorRay::App {

namespace {

using RmlKit::UI::ReportLevel;
using Scene::TextureSlot;

constexpr std::array imageFilters{
    RmlKit::UI::FileFilter{"Images", "png;jpg;jpeg;tga;bmp;hdr;exr"},
};
constexpr std::array<std::string_view, 3> alphaModeNames{"opaque", "mask", "blend"};
constexpr std::array<std::string_view, 4> channelNames{"r", "g", "b", "a"};

struct MaterialField {
    const char* name;
    std::variant<float Scene::Material::*, Core::float3 Scene::Material::*> member;
    std::optional<TextureSlot> slot;
};

constexpr std::array materialFields{
    MaterialField{"Base Color", &Scene::Material::baseColor, TextureSlot::BaseColor},
    MaterialField{"Metallic", &Scene::Material::metallic, TextureSlot::Metallic},
    MaterialField{"Roughness", &Scene::Material::roughness, TextureSlot::Roughness},
    MaterialField{"Normal", &Scene::Material::normalScale, TextureSlot::Normal},
    MaterialField{"IOR", &Scene::Material::ior, std::nullopt},
    MaterialField{"Alpha", &Scene::Material::alpha, TextureSlot::Alpha},
    MaterialField{"Specular IOR Level", &Scene::Material::specularIorLevel, TextureSlot::Specular},
    MaterialField{"Specular Tint", &Scene::Material::specularTint, std::nullopt},
    MaterialField{"Transmission", &Scene::Material::transmission, TextureSlot::Transmission},
    MaterialField{"Anisotropic", &Scene::Material::anisotropic, std::nullopt},
    MaterialField{"Anisotropic Rotation", &Scene::Material::anisotropicRotation, std::nullopt},
    MaterialField{"Coat", &Scene::Material::coat, TextureSlot::Coat},
    MaterialField{"Coat Roughness", &Scene::Material::coatRoughness, TextureSlot::CoatRoughness},
    MaterialField{"Coat Normal", &Scene::Material::coatNormalScale, TextureSlot::CoatNormal},
    MaterialField{"Sheen Color", &Scene::Material::sheenColor, std::nullopt},
    MaterialField{"Sheen Roughness", &Scene::Material::sheenRoughness, std::nullopt},
    MaterialField{"Emission Color", &Scene::Material::emissionColor, TextureSlot::Emission},
    MaterialField{"Emission Strength", &Scene::Material::emissionStrength, std::nullopt},
    MaterialField{"Thin Film Thickness", &Scene::Material::thinFilmThickness, std::nullopt},
    MaterialField{"Thin Film IOR", &Scene::Material::thinFilmIor, std::nullopt},
    MaterialField{"Dispersion", &Scene::Material::dispersion, std::nullopt},
};

struct Rgb {
    float r, g, b;
};

struct Hsv {
    float hue, saturation, value;
};

Rgb rgbOf(const Hsv hsv) {
    const auto channel = [&](const float n) {
        const float k = std::fmod(n + hsv.hue / 60.0f, 6.0f);
        return hsv.value - hsv.value * hsv.saturation * std::clamp(std::min(k, 4.0f - k), 0.0f, 1.0f);
    };
    return {channel(5.0f), channel(3.0f), channel(1.0f)};
}

Hsv hsvOf(const Rgb rgb) {
    const float high = std::max({rgb.r, rgb.g, rgb.b});
    const float range = high - std::min({rgb.r, rgb.g, rgb.b});
    float hue = 0.0f;
    if (range > 0.0f && high == rgb.r)
        hue = 60.0f * std::fmod((rgb.g - rgb.b) / range + 6.0f, 6.0f);
    else if (range > 0.0f && high == rgb.g)
        hue = 60.0f * ((rgb.b - rgb.r) / range + 2.0f);
    else if (range > 0.0f)
        hue = 60.0f * ((rgb.r - rgb.g) / range + 4.0f);
    return {hue, high > 0.0f ? range / high : 0.0f, high};
}

Rml::String hexOf(const Rgb rgb) {
    const auto byte = [](const float channel) { return static_cast<int>(std::lround(std::clamp(channel, 0.0f, 1.0f) * 255.0f)); };
    return std::format("#{:02x}{:02x}{:02x}", byte(rgb.r), byte(rgb.g), byte(rgb.b));
}

void showColor(MaterialParameterRow& row) {
    const Rgb rgb{std::max<float>(row.x, 0.0f), std::max<float>(row.y, 0.0f), std::max<float>(row.z, 0.0f)};
    const Hsv hsv = hsvOf(rgb);
    if (hsv.saturation > 0.0f)
        row.hue = static_cast<float>(std::lround(hsv.hue));
    row.saturation = static_cast<float>(std::lround(hsv.saturation * 100.0f));
    row.value = static_cast<float>(std::lround(hsv.value * 100.0f));
    row.color = hexOf(rgb);
    row.hueColor = hexOf(rgbOf({hsv.saturation > 0.0f ? hsv.hue : float(row.hue), 1.0f, 1.0f}));
}

MaterialParameterRow makeRow(const MaterialField& field, const Scene::Material& material, const Scene::Scene& scene) {
    MaterialParameterRow row;
    row.name = field.name;
    if (const auto* scalar = std::get_if<float Scene::Material::*>(&field.member)) {
        row.x = material.*(*scalar);
    } else {
        const Core::float3 color = material.*std::get<Core::float3 Scene::Material::*>(field.member);
        row.components = 3;
        row.x = color.x;
        row.y = color.y;
        row.z = color.z;
        showColor(row);
    }
    if (!field.slot)
        return row;
    row.textureSlot = true;
    const std::optional<Scene::MaterialTexture>& texture = material.texture(*field.slot);
    if (!texture)
        return row;
    const Scene::Texture* image = scene.find(texture->texture);
    row.texture = image ? std::filesystem::path(image->name).filename().string() : Rml::String("Missing");
    row.channel = Rml::String(channelNames[std::min<std::size_t>(texture->channel, channelNames.size() - 1)]);
    return row;
}

void writeRow(const MaterialField& field, const MaterialParameterRow& row, Scene::Material& material) {
    if (const auto* scalar = std::get_if<float Scene::Material::*>(&field.member)) {
        material.*(*scalar) = row.x;
        return;
    }
    material.*std::get<Core::float3 Scene::Material::*>(field.member) = {row.x, row.y, row.z};
}

}

MaterialController::MaterialController(RmlKit::UI::Model& model, AppModel& app, SDL_Window* window, Session::Session& session,
                                       ViewportElementInstancer& viewports)
    : model(model), app(app), window(window), session(session), viewports(viewports) {
    model.bindEvent("materialApply", &MaterialController::onApply, this);
    model.bindEvent("materialColor", &MaterialController::onColor, this);
    model.bindEvent("materialHex", &MaterialController::onHex, this);
    model.bindEvent("materialRename", &MaterialController::onRename, this);
    model.bindEvent("materialAlphaMode", &MaterialController::onAlphaMode, this);
    model.bindEvent("materialTexture", &MaterialController::onTexture, this);
    model.bindEvent("materialTextureClear", &MaterialController::onTextureClear, this);
    model.bindEvent("materialTextureChannel", &MaterialController::onTextureChannel, this);
}

void MaterialController::edit(const Scene::MaterialHandle edited) {
    if (edited != material)
        app.materialEditor.pickerParameter = -1;
    material = edited;
    refresh();
}

void MaterialController::edit(const std::size_t sceneIndex) {
    const Core::SlotMap<Scene::Material>& materials = session.getScene().getMaterials();
    if (sceneIndex >= materials.size())
        RmlKit::UI::fatal(std::format("the scene has no material {}", sceneIndex));
    edit(materials.keyAt(static_cast<std::uint32_t>(sceneIndex)));
}

void MaterialController::refresh() {
    const Scene::Scene& scene = session.getScene();
    MaterialEditorModel& editor = app.materialEditor;
    editor.parameters.clear();
    const Scene::Material* current = scene.find(material);
    if (!current) {
        material = {};
        editor.sceneIndex = -1;
        editor.name.clear();
        model.dirty("material");
        return;
    }
    editor.sceneIndex = static_cast<int>(scene.getMaterials().denseIndex(material));
    editor.name = current->name;
    editor.alphaMode = Rml::String(alphaModeNames[static_cast<std::size_t>(current->alphaMode)]);
    for (const MaterialField& field : materialFields)
        editor.parameters.push_back(makeRow(field, *current, scene));
    editor.parameterIndex = std::clamp(editor.parameterIndex, 0, static_cast<int>(editor.parameters.size()) - 1);
    model.dirty("material");
}

std::size_t MaterialController::parameterArgument(const Rml::VariantList& arguments, const char* event) const {
    return RmlKit::UI::Model::indexArgument(arguments, event, app.materialEditor.parameters.size());
}

void MaterialController::onApply(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    write(parameterArgument(arguments, "materialApply"));
}

void MaterialController::onColor(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    const std::size_t parameter = parameterArgument(arguments, "materialColor");
    MaterialParameterRow& row = app.materialEditor.parameters[parameter];
    const Rgb rgb = rgbOf({std::fmod(std::fmod(float(row.hue), 360.0f) + 360.0f, 360.0f), std::clamp<float>(row.saturation, 0.0f, 100.0f) / 100.0f,
                           std::max<float>(row.value, 0.0f) / 100.0f});
    row.x = rgb.r;
    row.y = rgb.g;
    row.z = rgb.b;
    write(parameter);
}

//! The colour picker writes the row's colour as #rrggbb.
void MaterialController::onHex(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t parameter = parameterArgument(arguments, "materialHex");
    MaterialParameterRow& row = app.materialEditor.parameters[parameter];
    unsigned int packed = 0;
    if (row.color.size() != 7 || std::sscanf(row.color.c_str(), "#%6x", &packed) != 1)
        return;
    row.x = static_cast<float>((packed >> 16) & 0xff) / 255.0f;
    row.y = static_cast<float>((packed >> 8) & 0xff) / 255.0f;
    row.z = static_cast<float>(packed & 0xff) / 255.0f;
    write(parameter);
}

void MaterialController::write(const std::size_t parameter) {
    const Scene::Material* current = session.getScene().find(material);
    if (!current)
        return;
    MaterialParameterRow& row = app.materialEditor.parameters[parameter];
    Scene::Material changedMaterial = *current;
    writeRow(materialFields[parameter], row, changedMaterial);
    if (row.components >= 3)
        showColor(row);
    session.getScene().setMaterial(material, changedMaterial);
    changed();
}

void MaterialController::onRename(Rml::DataModelHandle, Rml::Event& event, const Rml::VariantList& arguments) {
    if (!FieldCommit::isCommit(event))
        return;
    RmlKit::UI::Model::requireNoArguments(arguments, "materialRename");
    const Scene::Material* current = session.getScene().find(material);
    if (!current || app.materialEditor.name.empty() || app.materialEditor.name == current->name)
        return;
    Scene::Material renamed = *current;
    renamed.name = app.materialEditor.name;
    session.getScene().setMaterial(material, renamed);
    const std::size_t index = session.getScene().getMaterials().denseIndex(material);
    if (index < app.outliner.materials.size())
        app.outliner.materials[index] = renamed.name;
    for (MaterialLink& link : app.inspector.materials)
        if (link.sceneIndex == static_cast<int>(index))
            link.name = renamed.name;
    model.dirty("outliner");
    model.dirty("inspector");
    changed();
}

void MaterialController::onAlphaMode(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    RmlKit::UI::Model::requireNoArguments(arguments, "materialAlphaMode");
    const Scene::Material* current = session.getScene().find(material);
    const auto found = std::ranges::find(alphaModeNames, std::string_view(app.materialEditor.alphaMode));
    if (!current || found == alphaModeNames.end())
        return;
    const auto mode = static_cast<Scene::AlphaMode>(found - alphaModeNames.begin());
    if (mode == current->alphaMode)
        return;
    Scene::Material changedMaterial = *current;
    changedMaterial.alphaMode = mode;
    session.getScene().setMaterial(material, changedMaterial);
    changed();
}

void MaterialController::onTexture(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t parameter = parameterArgument(arguments, "materialTexture");
    const std::optional<TextureSlot> slot = materialFields[parameter].slot;
    if (!slot)
        return;
    RmlKit::UI::FileDialog::openFile(window, imageFilters, {}, [this, edited = material, slot = *slot](const std::filesystem::path& path) {
        if (path.empty())
            return;
        if (auto loaded = session.loadMaterialTexture(edited, slot, path); !loaded) {
            model.report(ReportLevel::Error, "material", std::format("NoorRay cannot load {}: {}", path.filename().string(), loaded.error().message));
            return;
        }
        changed();
    });
}

void MaterialController::onTextureClear(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t parameter = parameterArgument(arguments, "materialTextureClear");
    if (const std::optional<TextureSlot> slot = materialFields[parameter].slot) {
        session.clearMaterialTexture(material, *slot);
        changed();
    }
}

void MaterialController::onTextureChannel(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments) {
    const std::size_t parameter = parameterArgument(arguments, "materialTextureChannel");
    const std::optional<TextureSlot> slot = materialFields[parameter].slot;
    const Scene::Material* current = session.getScene().find(material);
    const auto found = std::ranges::find(channelNames, std::string_view(app.materialEditor.parameters[parameter].channel));
    if (!slot || !current || !current->texture(*slot) || found == channelNames.end())
        return;
    const auto channel = static_cast<std::uint32_t>(found - channelNames.begin());
    if (channel == current->texture(*slot)->channel)
        return;
    Scene::Material changedMaterial = *current;
    changedMaterial.texture(*slot)->channel = channel;
    session.getScene().setMaterial(material, changedMaterial);
    changed();
}

void MaterialController::changed() {
    viewports.invalidate();
    refresh();
}

}
