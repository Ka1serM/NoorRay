#pragma once

#include <cstddef>
#include <functional>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Types.h>
#include <SDL3/SDL_video.h>

#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Model;
}

namespace NoorRay::App {

class AppModel;
class ViewportElementInstancer;

class MaterialController {
public:
    MaterialController(RmlKit::UI::Model& model, AppModel& app, SDL_Window* window, Session::Session& session, ViewportElementInstancer& viewports);
    MaterialController(const MaterialController&) = delete;
    MaterialController& operator=(const MaterialController&) = delete;

    void edit(Scene::MaterialHandle material);
    void edit(std::size_t sceneIndex);
    //! Shows the edited material again, or nothing when the scene no longer has it.
    void refresh();
    Scene::MaterialHandle getMaterial() const { return material; }
    //! Gives the edited material a node tree with the same result as its fields and compiles it; false when it has a tree already.
    bool useNodes();

    //! Called when a material name changes, so the drawer shows it.
    std::function<void()> renamed;

private:
    void onApply(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onColor(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onHex(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onRename(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onAlphaMode(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onTexture(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onTextureClear(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);
    void onTextureChannel(Rml::DataModelHandle, Rml::Event&, const Rml::VariantList& arguments);

    std::size_t parameterArgument(const Rml::VariantList& arguments, const char* event) const;
    void write(std::size_t parameter);
    void changed();

    RmlKit::UI::Model& model;
    AppModel& app;
    SDL_Window* window;
    Session::Session& session;
    ViewportElementInstancer& viewports;
    Scene::MaterialHandle material;
};

}
