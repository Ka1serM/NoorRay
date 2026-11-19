#include "NoorRay.App/UI/Controls/ViewportElement.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <RmlUi/Core/ComputedValues.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Event.h>
#include <RmlUi/Core/Input.h>
#include <RmlUi/Core/MeshUtilities.h>
#include <RmlUi/Core/RenderBox.h>
#include <RmlUi/Core/RenderManager.h>
#include <SDL3/SDL_mouse.h>

#include "RmlKit/UI/Backend/Renderer.h"

namespace NoorRay::App {

namespace {

constexpr int leftButton = 0;
constexpr int rightButton = 1;
constexpr int middleButton = 2;
constexpr float orbitSpeed = 0.006f;
constexpr float dollySpeed = 0.05f;
constexpr float clickSlop = 3.0f;
constexpr float maxFlyStep = 0.1f;
//! A live window resize changes the size on every frame; the render targets follow at most this often, so the old ones are not dropped and allocated again each
//! frame.
constexpr std::chrono::milliseconds resizeSettle{150};
//! The viewport returns to full resolution once the camera has rested this long, so a pause between scroll steps keeps the low resolution.
constexpr std::chrono::milliseconds motionSettle{200};
constexpr Rml::EventId mouseEvents[] = {Rml::EventId::Mousedown,   Rml::EventId::Mousemove, Rml::EventId::Mouseup, Rml::EventId::Mouseout,
                                        Rml::EventId::Mousescroll, Rml::EventId::Dblclick,  Rml::EventId::Click};
constexpr const char* fileDropEvent = "filedrop";

enum FlyKey : std::uint32_t {
    flyForward = 1u << 0,
    flyBack = 1u << 1,
    flyLeft = 1u << 2,
    flyRight = 1u << 3,
    flyDown = 1u << 4,
    flyUp = 1u << 5,
    flyFast = 1u << 6,
};

std::uint32_t flyKeyOf(const int key) {
    switch (key) {
    case Rml::Input::KI_W:
        return flyForward;
    case Rml::Input::KI_S:
        return flyBack;
    case Rml::Input::KI_A:
        return flyLeft;
    case Rml::Input::KI_D:
        return flyRight;
    case Rml::Input::KI_Q:
        return flyDown;
    case Rml::Input::KI_E:
        return flyUp;
    case Rml::Input::KI_LSHIFT:
    case Rml::Input::KI_RSHIFT:
        return flyFast;
    default:
        return 0;
    }
}

Uint8 toSdlButton(const int button) {
    switch (button) {
    case rightButton:
        return SDL_BUTTON_RIGHT;
    case middleButton:
        return SDL_BUTTON_MIDDLE;
    default:
        return SDL_BUTTON_LEFT;
    }
}

Core::float3 flyDirection(const std::uint32_t held) {
    const auto axis = [&](const std::uint32_t positive, const std::uint32_t negative) {
        return ((held & positive) != 0 ? 1.0f : 0.0f) - ((held & negative) != 0 ? 1.0f : 0.0f);
    };
    const Core::float3 direction{axis(flyRight, flyLeft), axis(flyUp, flyDown), axis(flyBack, flyForward)};
    return direction * ((held & flyFast) != 0 ? 4.0f : 1.0f);
}

}

ViewportElement::ViewportElement(const Rml::String& tag, ViewportElementInstancer& instancer)
    : Rml::Element(tag), instancer(instancer), maximizer(*this, *this) {
    for (const Rml::EventId event : mouseEvents)
        AddEventListener(event, this);
    AddEventListener(fileDropEvent, this);
    SetClass("file-drop", true);
}

ViewportElement::~ViewportElement() {
    for (const Rml::EventId event : mouseEvents)
        RemoveEventListener(event, this);
    RemoveEventListener(fileDropEvent, this);
    if (!keyContext)
        return;
    keyContext->RemoveEventListener("keydown", &keys, true);
    keyContext->RemoveEventListener("keyup", &keys, true);
    keyContext->RemoveEventListener("mousemove", &pointer);
    keyContext->RemoveEventListener("mouseup", &pointer);
}

void ViewportElement::ProcessEvent(Rml::Event& event) {
    if (maximizer.handleKey(event))
        return;
    if (event.GetType() == fileDropEvent) {
        instancer.getEvents().fileDrop(std::filesystem::path(event.GetParameter<Rml::String>("path", "")));
        return;
    }
    processMouse(event);
}

void ViewportElement::processMouse(Rml::Event& event) {
    const Rml::Vector2f mouse(event.GetParameter<float>("mouse_x", 0), event.GetParameter<float>("mouse_y", 0));
    const int button = event.GetParameter<int>("button", 0);
    const float height = std::max(GetBox().GetSize(Rml::BoxArea::Content).y, 1.0f);
    const Rml::Element* target = event.GetTargetElement();
    switch (event.GetId()) {
    case Rml::EventId::Mousedown: {
        pressedAt = mouse;
        movedSincePress = false;
        pendingDrag.reset();
        if (target != this) {
            const std::optional<ViewDrag> tool = ViewGizmo::dragOf(target);
            if (tool && button == leftButton && ViewGizmo::axisOf(target) >= 0)
                pendingDrag = tool;
            else if (tool && button == leftButton)
                instancer.beginDrag(*tool, button, height);
            pendingButton = button;
            return;
        }
        pendingButton = button;
        if (button == leftButton && event.GetParameter<int>("shift_key", 0) == 0 && pressOverlay(mouse))
            return;
        if (button == leftButton)
            pendingDrag = event.GetParameter<int>("shift_key", 0) != 0 ? ViewDrag::Pan : ViewDrag::Orbit;
        if (button == middleButton)
            pendingDrag = ViewDrag::Pan;
        if (button == rightButton) {
            instancer.beginDrag(ViewDrag::Look, button, height);
            lastFly = std::chrono::steady_clock::now();
        }
        return;
    }
    case Rml::EventId::Mousemove:
        if (target == this && !pendingDrag && !instancer.getDrag())
            hoverOverlay(mouse);
        if (!pendingDrag || std::abs(mouse.x - pressedAt.x) + std::abs(mouse.y - pressedAt.y) <= clickSlop)
            return;
        movedSincePress = true;
        instancer.beginDrag(*std::exchange(pendingDrag, std::nullopt), pendingButton, height);
        instancer.moveDrag(mouse - pressedAt);
        return;
    case Rml::EventId::Mouseup:
        if (button == leftButton && pendingDrag == ViewDrag::Orbit && target == this && !movedSincePress)
            pendingPick = toPixel(mouse);
        pendingDrag.reset();
        return;
    case Rml::EventId::Click:
        if (const int axis = ViewGizmo::axisOf(target); axis >= 0 && !movedSincePress) {
            instancer.getCamera().snap(axis);
            requestDraw();
        }
        return;
    //! Before the mouse is grabbed, RmlUi sends no mouse event outside the element, so a press that has not become a drag ends there.
    case Rml::EventId::Mouseout:
        if (target == this)
            pendingDrag.reset();
        if (target == this && !overlay.getDragged() && std::exchange(hovered, std::nullopt))
            requestDraw();
        return;
    case Rml::EventId::Mousescroll:
        instancer.getCamera().zoom(event.GetParameter<float>("wheel_delta_y", 0));
        instancer.applyCamera();
        event.StopPropagation();
        return;
    case Rml::EventId::Dblclick:
        if (target == this)
            instancer.frameScene();
        return;
    default:
        return;
    }
}

void ViewportElement::processKey(Rml::Event& event) {
    if (!IsVisible(true))
        return;
    const int key = event.GetParameter<int>("key_identifier", Rml::Input::KI_UNKNOWN);
    if (const Rml::Element* target = event.GetTargetElement(); target && (target->GetTagName() == "input" || target->GetTagName() == "textarea"))
        return;
    const bool pressed = event.GetType() == "keydown";
    if (pressed && key == Rml::Input::KI_F && IsPseudoClassSet("hover")) {
        instancer.frameScene();
        event.StopPropagation();
        return;
    }
    const std::uint32_t flyKey = flyKeyOf(key);
    if (flyKey == 0 || instancer.getDrag() != ViewDrag::Look)
        return;
    heldKeys = pressed ? heldKeys | flyKey : heldKeys & ~flyKey;
    event.StopPropagation();
    requestDraw();
}

void ViewportElement::flyStep() {
    const auto now = std::chrono::steady_clock::now();
    const float seconds = std::min(std::chrono::duration<float>(now - std::exchange(lastFly, now)).count(), maxFlyStep);
    if (instancer.getDrag() != ViewDrag::Look)
        heldKeys = 0;
    if ((heldKeys & ~flyFast) == 0)
        return;
    instancer.getCamera().fly(flyDirection(heldKeys), seconds);
    instancer.applyCamera();
}

std::optional<Rml::Vector2i> ViewportElement::toPixel(const Rml::Vector2f mouse) {
    const Rml::Vector2f offset = mouse - GetAbsoluteOffset(Rml::BoxArea::Content);
    const Rml::Vector2f size = GetBox().GetSize(Rml::BoxArea::Content);
    const Render::Renderer& renderer = instancer.getSession().getRenderer();
    if (offset.x < 0 || offset.y < 0 || offset.x >= size.x || offset.y >= size.y)
        return std::nullopt;
    return Rml::Vector2i(static_cast<int>(offset.x / size.x * renderer.getWidth()), static_cast<int>(offset.y / size.y * renderer.getHeight()));
}

void ViewportElement::requestDraw() {
    if (Rml::Context* context = GetContext())
        context->RequestNextUpdate(0);
}

bool ViewportElement::isCameraMoving() {
    const auto now = std::chrono::steady_clock::now();
    const Scene::Camera* active = instancer.getSession().getScene().findActiveCamera();
    const std::optional<Core::Transform> transform = active ? std::optional(active->transform) : std::nullopt;
    if (transform != lastCameraTransform) {
        if (lastCameraTransform && transform)
            lastCameraMove = now;
        lastCameraTransform = transform;
    }
    return instancer.getDrag() || (heldKeys & ~flyFast) != 0 || instancer.getCamera().isAnimating() || now - lastCameraMove < motionSettle;
}

void ViewportElement::resizeTarget() {
    const Rml::Vector2f size = GetBox().GetSize(Rml::BoxArea::Content);
    const std::uint32_t divisor = isCameraMoving() ? std::max(instancer.getMotionDivisor(), 1u) : 1u;
    const auto width = static_cast<std::uint32_t>(std::round(size.x / static_cast<float>(divisor)));
    const auto height = static_cast<std::uint32_t>(std::round(size.y / static_cast<float>(divisor)));
    Session::Session& session = instancer.getSession();
    const Render::Renderer& renderer = session.getRenderer();
    if (divisor > 1)
        requestDraw();
    if (width == 0 || height == 0 || (renderer.getWidth() == width && renderer.getHeight() == height))
        return;
    const auto now = std::chrono::steady_clock::now();
    //! Only a change of the element size waits to settle; entering or leaving the motion resolution resizes at once.
    const bool divisorChanged = std::exchange(shownDivisor, divisor) != divisor;
    if (!divisorChanged && renderer.getDisplay() && lastResize && now - *lastResize < resizeSettle) {
        requestDraw();
        return;
    }
    lastResize = now;
    if (auto resized = session.resize(width, height); !resized) {
        failed = true;
        instancer.getEvents().error(resized.error().message);
    }
}

void ViewportElement::OnUpdate() {
    maximizer.update();
    if (!keyContext && (keyContext = GetContext())) {
        keyContext->AddEventListener("keydown", &keys, true);
        keyContext->AddEventListener("keyup", &keys, true);
        keyContext->AddEventListener("mousemove", &pointer);
        keyContext->AddEventListener("mouseup", &pointer);
        instancer.setContext(keyContext);
    }
    if (!IsVisible(true))
        return;
    flyStep();
    const bool ticking = instancer.getEvents().tick && instancer.getEvents().tick();
    if (instancer.getCamera().animate())
        instancer.applyCamera();
    Session::Session& session = instancer.getSession();
    if (const Scene::Camera* active = session.getScene().findActiveCamera())
        gizmo.sync(*this, active->transform);
    if (std::optional<Session::PickResult> picked = session.takePick())
        instancer.getEvents().pick(*picked);
    if (drawnGeneration != instancer.getGeneration())
        failed = false;
    if (failed)
        return;
    resizeTarget();
    if (failed)
        return;
    const bool changed = std::exchange(drawnGeneration, instancer.getGeneration()) != instancer.getGeneration();
    if (changed)
        instancer.fitClipRange();
    if (changed || !session.getRenderer().isComplete() || pendingPick) {
        NoorRHI::CommandList& list = instancer.getRenderer().record();
        if (auto recorded = session.record(list); !recorded) {
            failed = true;
            instancer.getEvents().error(recorded.error().message);
            return;
        }
        if (const std::optional<Rml::Vector2i> pick = std::exchange(pendingPick, std::nullopt))
            session.requestPick(list, static_cast<std::uint32_t>(pick->x), static_cast<std::uint32_t>(pick->y));
        instancer.getEvents().frame();
    }
    if (ticking || !session.getRenderer().isComplete() || instancer.getDrag() == ViewDrag::Look || instancer.getCamera().isAnimating())
        requestDraw();
}

void ViewportElement::showDisplay(Rml::RenderManager& renderManager) {
    const NoorRHI::Image& display = instancer.getSession().getRenderer().getDisplay();
    if (!display || display.getBackend() == shownDisplay)
        return;
    shownDisplay = display.getBackend();
    geometryDirty = true;
    RmlKit::UI::Renderer& renderer = instancer.getRenderer();
    texture = renderManager.MakeCallbackTexture(
        [&renderer, image = display,
         size = Rml::Vector2i(static_cast<int>(display.getWidth()), static_cast<int>(display.getHeight()))](const Rml::CallbackTextureInterface& callback) {
            callback.SetTextureHandle(renderer.addTexture(image, RmlKit::UI::TextureMode::Premultiplied), size);
            return true;
        });
}

void ViewportElement::OnRender() {
    Rml::RenderManager* renderManager = GetRenderManager();
    if (!renderManager)
        return;
    showDisplay(*renderManager);
    if (!texture)
        return;
    if (std::exchange(geometryDirty, false))
        generateGeometry(*renderManager);
    geometry.Render(GetAbsoluteOffset(Rml::BoxArea::Border), texture);
    renderOverlay(*renderManager);
}

GizmoMode ViewportElement::gizmoMode() {
    ViewportEvents& events = instancer.getEvents();
    return events.tool ? events.tool() : GizmoMode::Move;
}

float ViewportElement::getDensity() {
    const Rml::Context* context = GetContext();
    return context ? context->GetDensityIndependentPixelRatio() : 1.0f;
}

Rml::Vector2f ViewportElement::toContent(const Rml::Vector2f mouse) {
    return mouse - GetAbsoluteOffset(Rml::BoxArea::Content);
}

std::optional<OverlayView> ViewportElement::overlayView() {
    const Scene::Camera* active = instancer.getSession().getScene().findActiveCamera();
    const Rml::Vector2f size = GetBox().GetSize(Rml::BoxArea::Content);
    if (!active || size.x <= 0.0f || size.y <= 0.0f)
        return std::nullopt;
    return OverlayView(*active, size);
}

bool ViewportElement::pressOverlay(const Rml::Vector2f mouse) {
    const std::optional<OverlayView> view = overlayView();
    ViewportEvents& events = instancer.getEvents();
    if (!view || !events.selection)
        return false;
    const Scene::Scene& scene = instancer.getSession().getScene();
    const Rml::Vector2f local = toContent(mouse);
    const Selection selection = events.selection();
    if (const std::optional<Core::Transform> target = transformOf(scene, selection)) {
        const GizmoMode mode = gizmoMode();
        if (const std::optional<GizmoPart> part = overlay.hitGizmo(*view, *target, local, getDensity(), mode);
            part && overlay.beginDrag(*view, *part, *target, local, getDensity(), mode)) {
            requestDraw();
            return true;
        }
    }
    const std::optional<Selection> icon = overlay.hitIcon(scene, *view, local, getDensity());
    if (!icon || !events.select)
        return false;
    events.select(*icon);
    requestDraw();
    return true;
}

void ViewportElement::hoverOverlay(const Rml::Vector2f mouse) {
    const std::optional<OverlayView> view = overlayView();
    ViewportEvents& events = instancer.getEvents();
    std::optional<GizmoPart> part;
    if (view && events.selection)
        if (const std::optional<Core::Transform> target = transformOf(instancer.getSession().getScene(), events.selection()))
            part = overlay.hitGizmo(*view, *target, toContent(mouse), getDensity(), gizmoMode());
    if (std::exchange(hovered, part) != part)
        requestDraw();
}

void ViewportElement::processGizmoDrag(Rml::Event& event) {
    if (!overlay.getDragged())
        return;
    if (event.GetId() == Rml::EventId::Mouseup) {
        overlay.endDrag();
        requestDraw();
        return;
    }
    const std::optional<OverlayView> view = overlayView();
    ViewportEvents& events = instancer.getEvents();
    if (!view || !events.selection)
        return;
    const Rml::Vector2f mouse(event.GetParameter<float>("mouse_x", 0), event.GetParameter<float>("mouse_y", 0));
    const std::optional<Core::Transform> moved = overlay.drag(*view, toContent(mouse));
    if (!moved)
        return;
    setTransform(instancer.getSession().getScene(), events.selection(), *moved);
    instancer.invalidate();
    if (events.moved)
        events.moved();
}

void ViewportElement::renderOverlay(Rml::RenderManager& renderManager) {
    const std::optional<OverlayView> view = overlayView();
    if (!view)
        return;
    ViewportEvents& events = instancer.getEvents();
    OverlayKey key{instancer.getGeneration(),
                   events.selection ? events.selection() : Selection{},
                   overlay.getDragged() ? overlay.getDragged() : hovered,
                   GetBox().GetSize(Rml::BoxArea::Content),
                   getDensity(),
                   gizmoMode()};
    if (const int cell = ViewportOverlay::iconCell(key.density); cell != iconAtlasCell) {
        iconAtlasCell = cell;
        iconAtlas = renderManager.MakeCallbackTexture([cell](const Rml::CallbackTextureInterface& callback) {
            return callback.GenerateTexture(ViewportOverlay::paintIconAtlas(cell), {cell * ViewportOverlay::iconCount, cell});
        });
    }
    if (overlayKey != key) {
        const Scene::Scene& scene = instancer.getSession().getScene();
        iconGeometry = renderManager.MakeGeometry(overlay.buildIcons(scene, *view, key.density));
        overlayGeometry = renderManager.MakeGeometry(overlay.build(scene, *view, key.selection, key.highlighted, key.density, key.mode));
        overlayKey = std::move(key);
    }
    const Rml::Vector2f offset = GetAbsoluteOffset(Rml::BoxArea::Content);
    const Rml::Rectanglei previous = renderManager.GetScissorRegion();
    const Rml::Rectanglei content = Rml::Rectanglei::FromPositionSize(Rml::Vector2i(offset), Rml::Vector2i(overlayKey->size));
    renderManager.SetScissorRegion(content.IntersectIfValid(previous));
    if (iconGeometry)
        iconGeometry.Render(offset, iconAtlas);
    if (overlayGeometry)
        overlayGeometry.Render(offset);
    if (previous.Valid())
        renderManager.SetScissorRegion(previous);
    else
        renderManager.DisableScissorRegion();
}

void ViewportElement::OnResize() {
    geometryDirty = true;
    requestDraw();
}

void ViewportElement::generateGeometry(Rml::RenderManager& renderManager) {
    const Rml::RenderBox box = GetRenderBox(Rml::BoxArea::Content);
    //! Until the render targets follow a resize, the old image covers the box in its own aspect, cropped at the centre rather than stretched.
    const Render::Renderer& renderer = instancer.getSession().getRenderer();
    const Rml::Vector2f fill = box.GetFillSize();
    Rml::Vector2f extent(1.0f, 1.0f);
    if (renderer.getWidth() > 0 && renderer.getHeight() > 0 && fill.x > 0.0f && fill.y > 0.0f) {
        const float ratio = (fill.x / fill.y) / (float(renderer.getWidth()) / float(renderer.getHeight()));
        extent = ratio > 1.0f ? Rml::Vector2f(1.0f, 1.0f / ratio) : Rml::Vector2f(ratio, 1.0f);
    }
    const Rml::Vector2f start = (Rml::Vector2f(1.0f, 1.0f) - extent) * 0.5f;
    Rml::Mesh mesh;
    Rml::MeshUtilities::GenerateQuad(mesh, box.GetFillOffset(), fill, Rml::Colourb(255).ToPremultiplied(GetComputedValues().opacity()), start, start + extent);
    geometry = renderManager.MakeGeometry(std::move(mesh));
}

void ViewportElementInstancer::beginDrag(const ViewDrag started, const int rmlButton, const float viewportHeight) {
    if (drag)
        return;
    drag = started;
    dragButton = toSdlButton(rmlButton);
    dragHeight = std::max(viewportHeight, 1.0f);
    SDL_GetMouseState(&restoreX, &restoreY);
    relative = SDL_SetWindowRelativeMouseMode(window, true);
}

void ViewportElementInstancer::moveDrag(const Rml::Vector2f moved) {
    if (!drag)
        return;
    switch (*drag) {
    case ViewDrag::Orbit:
        camera.orbit(-moved.x * orbitSpeed, moved.y * orbitSpeed);
        break;
    case ViewDrag::Pan:
        camera.pan(-moved.x / dragHeight, moved.y / dragHeight);
        break;
    case ViewDrag::Look:
        camera.look(-moved.x * orbitSpeed, moved.y * orbitSpeed);
        break;
    case ViewDrag::Dolly:
        camera.zoom(moved.y * dollySpeed);
        break;
    }
    applyCamera();
}

void ViewportElementInstancer::endDrag() {
    drag.reset();
    if (!std::exchange(relative, false))
        return;
    SDL_SetWindowRelativeMouseMode(window, false);
    SDL_WarpMouseInWindow(window, restoreX, restoreY);
}

bool ViewportElementInstancer::handleEvent(const SDL_Event& event) {
    if (!drag)
        return false;
    switch (event.type) {
    case SDL_EVENT_MOUSE_MOTION: {
        //! RmlUi gets mouse positions in pixels, and SDL reports motion in window coordinates.
        const float density = SDL_GetWindowPixelDensity(window);
        moveDrag(Rml::Vector2f(event.motion.xrel, event.motion.yrel) * density);
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_UP:
        if (event.button.button == dragButton)
            endDrag();
        return false;
    case SDL_EVENT_WINDOW_FOCUS_LOST:
        endDrag();
        return false;
    default:
        return false;
    }
}

void ViewportElementInstancer::setMotionDivisor(const std::uint32_t divisor) {
    motionDivisor = std::max(divisor, 1u);
}

void ViewportElementInstancer::invalidate() {
    ++generation;
    if (context)
        context->RequestNextUpdate(0);
}

void ViewportElementInstancer::frameScene() {
    const Scene::Camera* active = session.getScene().findActiveCamera();
    camera.frame(session.getScene().computeBounds(), active ? active->verticalFov : 0.7f);
    applyCamera();
}

void ViewportElementInstancer::followActiveCamera() {
    if (const Scene::Camera* active = session.getScene().findActiveCamera())
        camera.follow(active->transform);
}

void ViewportElementInstancer::applyCamera() {
    Scene::Scene& scene = session.getScene();
    const Scene::Camera* active = scene.findActiveCamera();
    if (!active)
        return;
    Scene::Camera moved = *active;
    moved.previousTransform = moved.transform;
    moved.transform = camera.getTransform();
    Scene::fitClipRange(moved, scene.computeBounds());
    scene.setCamera(scene.getActiveCamera(), moved);
    invalidate();
}

void ViewportElementInstancer::fitClipRange() {
    Scene::Scene& scene = session.getScene();
    const Scene::Camera* active = scene.findActiveCamera();
    if (!active)
        return;
    Scene::Camera fitted = *active;
    Scene::fitClipRange(fitted, scene.computeBounds());
    if (fitted.near == active->near && fitted.far == active->far)
        return;
    fitted.previousTransform = fitted.transform;
    scene.setCamera(scene.getActiveCamera(), fitted);
}

Rml::ElementPtr ViewportElementInstancer::InstanceElement(Rml::Element*, const Rml::String& tag, const Rml::XMLAttributes&) {
    return Rml::ElementPtr(new ViewportElement(tag, *this));
}

void ViewportElementInstancer::ReleaseElement(Rml::Element* element) {
    delete element;
}

}
