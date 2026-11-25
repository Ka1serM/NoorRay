#pragma once

#include <filesystem>
#include <functional>
#include <string_view>
#include <vector>

#include <RmlUi/Core/Types.h>
#include <SDL3/SDL_video.h>

#include "NoorRay.App/Plugins/ViewportSource.h"
#include "NoorRay/Session/Session.h"
//! Hotkeys.h uses std::move_only_function, which libc++ lacks, without its fallback.
#include "RmlKit/Core/StdGaps.h"
#include "RmlKit/UI/Backend/SecondaryWindow.h"
#include "RmlKit/UI/Backend/Window.h"

namespace NoorRay::App {

class Plugin;

//! The size of the window that the Plugins menu opens for a plugin, in pixels at a scale of 1.
struct PluginWindowSize {
    int width = 720;
    int height = 640;
    int minimumWidth = 420;
    int minimumHeight = 360;
};

//! What the app gives its plugins; a plugin sees the app only through this.
class PluginHost {
public:
    virtual ~PluginHost() = default;

    virtual Session::Session& getSession() = 0;
    virtual SDL_Window* getWindow() const = 0;
    //! The window stays valid until the plugin's windowClosed returns for it.
    virtual RmlKit::UI::SecondaryWindow& openWindow(const RmlKit::UI::WindowOptions& options) = 0;
    //! The plugin's own window: its name is the title and its getWindowSize the size.
    virtual RmlKit::UI::SecondaryWindow& openWindow(const Plugin& plugin) = 0;
    //! Null gives the viewport back to the path tracer.
    virtual void setViewportSource(ViewportSource* source) = 0;
    //! The plugin's image or bounds changed, so the viewport draws again.
    virtual void redraw() = 0;
    //! The plugin added or removed objects of the scene, so the outliner lists them again.
    virtual void sceneChanged() = 0;
    virtual void frameViewport() = 0;
    //! Moves the viewport camera to this camera-to-world transform, which looks along -z with y up.
    virtual void moveCamera(const Core::Transform& transform, float verticalFov) = 0;
    virtual void reportError(const Rml::String& source, const Rml::String& message) = 0;
    virtual void reportDone(const Rml::String& source, const Rml::String& message) = 0;
};

//! A part of the app that links into it without the app knowing it; it registers itself with a static PluginRegistration.
class Plugin {
public:
    virtual ~Plugin() = default;

    virtual std::string_view getName() const = 0;
    virtual PluginWindowSize getWindowSize() const { return {}; }
    //! The host opens every plugin once after the device is ready.
    virtual void open(PluginHost& host) = 0;
    //! The user chose the plugin in the Plugins menu.
    virtual void show() = 0;
    //! Runs on every update of the viewport; true asks for another update.
    virtual bool update() = 0;
    //! Runs before the device closes; the plugin frees its GPU resources and stops its threads.
    virtual void close() = 0;
    virtual void windowClosed(RmlKit::UI::SecondaryWindow& window) = 0;
};

inline std::vector<Plugin*>& getPlugins() {
    static std::vector<Plugin*> plugins;
    return plugins;
}

struct PluginRegistration {
    explicit PluginRegistration(Plugin& plugin) { getPlugins().push_back(&plugin); }
};

}
