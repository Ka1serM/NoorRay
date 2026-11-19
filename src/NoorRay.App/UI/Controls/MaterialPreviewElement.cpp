#include "NoorRay.App/UI/Controls/MaterialPreviewElement.h"

#include <cmath>
#include <numbers>

#include <RmlUi/Core/ComputedValues.h>
#include <RmlUi/Core/Context.h>
#include <RmlUi/Core/Mesh.h>
#include <RmlUi/Core/RenderBox.h>
#include <RmlUi/Core/RenderManager.h>

#include "NoorRay/Core/Log.h"
#include "RmlKit/UI/Backend/Renderer.h"

namespace NoorRay::App {

namespace {

constexpr int discSegments = 64;
//! Many elements ask in one display frame; only the first of them within this time records a preview frame.
constexpr std::chrono::milliseconds pumpInterval{4};

}

MaterialPreviewElement::MaterialPreviewElement(const Rml::String& tag, MaterialPreviewInstancer& instancer) : Rml::Element(tag), instancer(instancer) {}

void MaterialPreviewElement::OnAttributeChange(const Rml::ElementAttributes& changed) {
    Rml::Element::OnAttributeChange(changed);
    if (changed.find("material") != changed.end())
        if (Rml::Context* context = GetContext())
            context->RequestNextUpdate(0);
}

void MaterialPreviewElement::OnUpdate() {
    if (!IsVisible(true))
        return;
    RmlKit::UI::Renderer* renderer = instancer.rendererOf(GetContext());
    if (!renderer)
        return;
    const Scene::Scene& scene = instancer.getSession().getScene();
    const int index = GetAttribute<int>("material", -1);
    const NoorRHI::Image* image = nullptr;
    if (index >= 0 && static_cast<std::uint32_t>(index) < scene.getMaterials().size())
        image = instancer.getPreviews().request(scene, scene.getMaterials().keyAt(static_cast<std::uint32_t>(index)));
    //! A changed material keeps showing its last preview until the new one is ready.
    SetClass("ready", image != nullptr || shownImage != nullptr);
    if (!image) {
        instancer.pump(*renderer);
        if (instancer.getPreviews().isBusy())
            if (Rml::Context* context = GetContext())
                context->RequestNextUpdate(0);
        return;
    }
    if (image->getBackend() == shownImage)
        return;
    shownImage = image->getBackend();
    const auto size = static_cast<int>(instancer.getPreviews().getSize());
    texture = GetRenderManager()->MakeCallbackTexture([renderer, image = *image, size](const Rml::CallbackTextureInterface& callback) {
        callback.SetTextureHandle(renderer->addTexture(image, RmlKit::UI::TextureMode::Premultiplied), {size, size});
        return true;
    });
}

void MaterialPreviewElement::OnRender() {
    Rml::RenderManager* renderManager = GetRenderManager();
    if (!renderManager || !texture || !IsClassSet("ready"))
        return;
    if (std::exchange(geometryDirty, false))
        generateGeometry(*renderManager);
    geometry.Render(GetAbsoluteOffset(Rml::BoxArea::Border), texture);
}

//! A disc over the sphere's outline, so the black around the sphere never shows and the tile keeps its own background.
void MaterialPreviewElement::generateGeometry(Rml::RenderManager& renderManager) {
    const Rml::RenderBox box = GetRenderBox(Rml::BoxArea::Content);
    const Rml::Vector2f fill = box.GetFillSize();
    const Rml::Vector2f center = box.GetFillOffset() + fill * 0.5f;
    const float side = std::min(fill.x, fill.y);
    const float silhouette = instancer.getPreviews().getSilhouette();
    //! One texel past the outline keeps the sphere's antialiased edge.
    const float uvRadius = 0.5f * silhouette + 1.0f / static_cast<float>(instancer.getPreviews().getSize());
    const float radius = side * uvRadius;
    const Rml::ColourbPremultiplied colour = Rml::Colourb(255).ToPremultiplied(GetComputedValues().opacity());
    Rml::Mesh mesh;
    mesh.vertices.push_back({center, colour, {0.5f, 0.5f}});
    for (int i = 0; i <= discSegments; ++i) {
        const float angle = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) / discSegments;
        const Rml::Vector2f direction(std::cos(angle), std::sin(angle));
        mesh.vertices.push_back({center + direction * radius, colour, Rml::Vector2f(0.5f, 0.5f) + direction * uvRadius});
        if (i > 0)
            mesh.indices.insert(mesh.indices.end(), {0, i, i + 1});
    }
    geometry = renderManager.MakeGeometry(std::move(mesh));
}

std::expected<void, Core::Error> MaterialPreviewInstancer::open() {
    return previews.open(session.getDevice(), session.getRenderer().getPipelines());
}

RmlKit::UI::Renderer* MaterialPreviewInstancer::rendererOf(const Rml::Context* context) const {
    const auto found = renderers.find(context);
    return found != renderers.end() ? found->second : nullptr;
}

void MaterialPreviewInstancer::pump(RmlKit::UI::Renderer& renderer) {
    const auto now = std::chrono::steady_clock::now();
    if (failed || now - lastPump < pumpInterval)
        return;
    lastPump = now;
    if (auto recorded = previews.record(renderer.record(), session.getScene()); !recorded) {
        failed = true;
        Core::Log::error("the material previews stop: {}", recorded.error().message);
    }
}

Rml::ElementPtr MaterialPreviewInstancer::InstanceElement(Rml::Element*, const Rml::String& tag, const Rml::XMLAttributes&) {
    return Rml::ElementPtr(new MaterialPreviewElement(tag, *this));
}

void MaterialPreviewInstancer::ReleaseElement(Rml::Element* element) {
    delete element;
}

}
