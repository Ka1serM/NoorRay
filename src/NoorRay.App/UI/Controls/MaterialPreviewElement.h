#pragma once

#include <chrono>
#include <cstdint>
#include <unordered_map>

#include <RmlUi/Core/CallbackTexture.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/Geometry.h>

#include "NoorRay/Session/MaterialPreviews.h"
#include "NoorRay/Session/Session.h"

namespace RmlKit::UI {
class Renderer;
}

namespace NoorRay::App {

class MaterialPreviewInstancer;

//! <materialpreview material="index"> shows the rendered preview of the scene material at that index; until it is ready it shows nothing.
class MaterialPreviewElement final : public Rml::Element {
public:
    MaterialPreviewElement(const Rml::String& tag, MaterialPreviewInstancer& instancer);

protected:
    void OnUpdate() override;
    void OnRender() override;
    void OnResize() override { geometryDirty = true; }
    void OnAttributeChange(const Rml::ElementAttributes& changed) override;

private:
    void generateGeometry(Rml::RenderManager& renderManager);

    MaterialPreviewInstancer& instancer;
    Rml::CallbackTexture texture;
    const void* shownImage = nullptr;
    Rml::Geometry geometry;
    bool geometryDirty = true;
};

//! Creates the <materialpreview> elements and owns the previews they share; each context draws with the renderer of its own window.
class MaterialPreviewInstancer final : public Rml::ElementInstancer {
public:
    explicit MaterialPreviewInstancer(Session::Session& session) : session(session) {}

    [[nodiscard]] std::expected<void, Core::Error> open();
    void addContext(Rml::Context* context, RmlKit::UI::Renderer& renderer) { renderers[context] = &renderer; }
    void removeContext(Rml::Context* context) { renderers.erase(context); }
    RmlKit::UI::Renderer* rendererOf(const Rml::Context* context) const;

    Session::Session& getSession() const { return session; }
    Session::MaterialPreviews& getPreviews() { return previews; }
    //! Records one preview frame into the renderer's list, at most once per display frame however many elements wait.
    void pump(RmlKit::UI::Renderer& renderer);

    Rml::ElementPtr InstanceElement(Rml::Element* parent, const Rml::String& tag, const Rml::XMLAttributes& attributes) override;
    void ReleaseElement(Rml::Element* element) override;

private:
    Session::Session& session;
    Session::MaterialPreviews previews;
    std::unordered_map<const Rml::Context*, RmlKit::UI::Renderer*> renderers;
    std::chrono::steady_clock::time_point lastPump;
    bool failed = false;
};

}
