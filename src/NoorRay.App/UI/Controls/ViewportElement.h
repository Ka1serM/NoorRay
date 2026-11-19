#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>

#include <RmlUi/Core/CallbackTexture.h>
#include <RmlUi/Core/Element.h>
#include <RmlUi/Core/ElementInstancer.h>
#include <RmlUi/Core/EventListener.h>
#include <RmlUi/Core/Geometry.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_video.h>

#include "NoorRay.App/UI/Controls/OrbitCamera.h"
#include "NoorRay.App/UI/Controls/ViewGizmo.h"
#include "NoorRay.App/UI/Controls/ViewportOverlay.h"
#include "NoorRay/Session/Session.h"
#include "RmlKit/UI/Controls/Maximizer.h"

namespace RmlKit::UI {
class Renderer;
}

namespace NoorRay::App {

struct ViewportEvents {
    std::function<void()> frame;
    std::function<void(const Session::PickResult&)> pick;
    std::function<void(std::filesystem::path)> fileDrop;
    std::function<void(const Rml::String&)> error;
    std::function<Selection()> selection;
    std::function<void(const Selection&)> select;
    //! The gizmo moved, scaled or rotated the selection.
    std::function<void()> moved;
    std::function<GizmoMode()> tool;
    //! Called on every update; true asks for another update, as playback does.
    std::function<bool()> tick;
};

class ViewportElementInstancer;

class ViewportElement final : public Rml::Element, public Rml::EventListener {
public:
    ViewportElement(const Rml::String& tag, ViewportElementInstancer& instancer);
    ~ViewportElement() override;

    void ProcessEvent(Rml::Event& event) override;

protected:
    void OnUpdate() override;
    void OnRender() override;
    void OnResize() override;

private:
    class Keys final : public Rml::EventListener {
    public:
        explicit Keys(ViewportElement& viewport) : viewport(viewport) {}
        void ProcessEvent(Rml::Event& event) override { viewport.processKey(event); }

    private:
        ViewportElement& viewport;
    };

    class Pointer final : public Rml::EventListener {
    public:
        explicit Pointer(ViewportElement& viewport) : viewport(viewport) {}
        void ProcessEvent(Rml::Event& event) override { viewport.processGizmoDrag(event); }

    private:
        ViewportElement& viewport;
    };

    void processKey(Rml::Event& event);
    void processGizmoDrag(Rml::Event& event);
    bool pressOverlay(Rml::Vector2f mouse);
    void hoverOverlay(Rml::Vector2f mouse);
    void renderOverlay(Rml::RenderManager& renderManager);
    std::optional<OverlayView> overlayView();
    Rml::Vector2f toContent(Rml::Vector2f mouse);
    float getDensity();
    GizmoMode gizmoMode();
    void processMouse(Rml::Event& event);
    void requestDraw();
    void flyStep();
    void resizeTarget();
    bool isCameraMoving();
    void showDisplay(Rml::RenderManager& renderManager);
    void generateGeometry(Rml::RenderManager& renderManager);
    std::optional<Rml::Vector2i> toPixel(Rml::Vector2f mouse);

    ViewportElementInstancer& instancer;
    Rml::CallbackTexture texture;
    const void* shownDisplay = nullptr;
    Rml::Geometry geometry;
    bool geometryDirty = true;
    std::uint64_t drawnGeneration = 0;
    std::optional<std::chrono::steady_clock::time_point> lastResize;
    //! When the active camera last moved, and where it was, so the viewport drops to the motion resolution while it moves.
    std::chrono::steady_clock::time_point lastCameraMove;
    std::optional<Core::Transform> lastCameraTransform;
    std::uint32_t shownDivisor = 1;
    bool failed = false;

    //! A left or middle press becomes a drag once the mouse moves past the click slop; until then it can still pick.
    std::optional<ViewDrag> pendingDrag;
    int pendingButton = 0;
    Rml::Vector2f pressedAt;
    bool movedSincePress = false;
    std::optional<Rml::Vector2i> pendingPick;
    std::uint32_t heldKeys = 0;
    std::chrono::steady_clock::time_point lastFly;

    ViewGizmo gizmo;
    ViewportOverlay overlay;
    Rml::Geometry overlayGeometry;
    Rml::Geometry iconGeometry;
    Rml::CallbackTexture iconAtlas;
    int iconAtlasCell = 0;
    std::optional<GizmoPart> hovered;
    //! What the overlay geometry shows; it is built again when one of these changes.
    struct OverlayKey {
        std::uint64_t generation = 0;
        Selection selection;
        std::optional<GizmoPart> highlighted;
        Rml::Vector2f size;
        float density = 0.0f;
        GizmoMode mode = GizmoMode::Move;
        bool operator==(const OverlayKey&) const = default;
    };
    std::optional<OverlayKey> overlayKey;
    RmlKit::UI::Maximizer maximizer;
    Keys keys{*this};
    Pointer pointer{*this};
    Rml::Context* keyContext = nullptr;
};

//! Creates the <viewport> elements and keeps what they share: the session, the navigation camera and the redraw generation.
class ViewportElementInstancer final : public Rml::ElementInstancer {
public:
    ViewportElementInstancer(RmlKit::UI::Renderer& renderer, Session::Session& session, SDL_Window* window)
        : renderer(renderer), session(session), window(window) {}

    RmlKit::UI::Renderer& getRenderer() const { return renderer; }
    Session::Session& getSession() const { return session; }
    OrbitCamera& getCamera() { return camera; }
    ViewportEvents& getEvents() { return events; }
    std::uint64_t getGeneration() const { return generation; }
    //! While the camera moves, the viewport renders at 1/divisor of its resolution; 1 always renders at full resolution.
    void setMotionDivisor(std::uint32_t divisor);
    std::uint32_t getMotionDivisor() const { return motionDivisor; }

    //! The scene, the settings or the camera changed: the next update records a new frame and a failed viewport tries again.
    void invalidate();
    void frameScene();
    void followActiveCamera();
    void applyCamera();
    //! Fits near and far of the active camera to the scene, which may have grown since the camera last moved.
    void fitClipRange();
    void setContext(Rml::Context* updated) { context = updated; }

    //! Grabs the mouse in relative mode, as the old viewport did; it returns to where the drag began when the button comes up.
    void beginDrag(ViewDrag drag, int rmlButton, float viewportHeight);
    //! Moves the camera by a drag of this many RmlUi pixels.
    void moveDrag(Rml::Vector2f moved);
    std::optional<ViewDrag> getDrag() const { return drag; }
    //! The motion of a grabbed mouse goes to the camera and not to RmlUi.
    bool handleEvent(const SDL_Event& event);

    Rml::ElementPtr InstanceElement(Rml::Element* parent, const Rml::String& tag, const Rml::XMLAttributes& attributes) override;
    void ReleaseElement(Rml::Element* element) override;

private:
    RmlKit::UI::Renderer& renderer;
    Session::Session& session;
    OrbitCamera camera;
    ViewportEvents events;
    void endDrag();

    SDL_Window* window;
    std::uint64_t generation = 1;
    std::uint32_t motionDivisor = 2;
    Rml::Context* context = nullptr;
    std::optional<ViewDrag> drag;
    Uint8 dragButton = 0;
    float dragHeight = 1.0f;
    float restoreX = 0.0f;
    float restoreY = 0.0f;
    bool relative = false;
};

}
