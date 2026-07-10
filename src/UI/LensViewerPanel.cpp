#include "LensViewerPanel.h"

#include <algorithm>
#include <cmath>
#include <imgui.h>

#include "Camera/CameraInstance.h"
#include "Camera/RealisticCamera.h"
#include "Scene/Scene.h"
#include "libross/foundation/physics/Wavelengths.h"

namespace {

class LensViewerRaytracer : public ross::SequentialRaytracer {
public:
    explicit LensViewerRaytracer(const ross::CameraLens& lens) : ross::SequentialRaytracer(lens) {}

    bool trace(const ross::Ray& filmRay, float wavelengthNm,
        std::vector<glm::vec2>& points, std::vector<LensViewerPanel::SurfaceHit>& hits) const
    {
        ross::Ray tracedRay{{filmRay.startPoint.x, filmRay.startPoint.y, -filmRay.startPoint.z},
            {filmRay.direction.x, filmRay.direction.y, -filmRay.direction.z}};
        points.push_back({-tracedRay.startPoint.z, tracedRay.startPoint.y});

        const auto& surfaces = cameraLens.surfaces;
        for (int i = static_cast<int>(surfaces.size()) - 1; i >= 0; --i) {
            const auto& surface = surfaces[i];
            if (surface.isAperture()) {
                if (exceedsAperture(surface, tracedRay))
                    return false;
                continue;
            }

            const auto intersection = intersect(surface, tracedRay);
            if (!intersection)
                return false;
            const glm::vec2 hitPoint{-intersection->point.z, intersection->point.y};
            points.push_back(hitPoint);
            const size_t hitIndex = hits.size();
            hits.push_back({hitPoint, glm::vec2(-intersection->normal.z, intersection->normal.y), {0.0f, 0.0f}});

            const float etaSurface = surface.material.getIor(wavelengthNm);
            const float etaPrevious = (i > 0 && surfaces[i - 1].material.getIor(wavelengthNm) != 0.0f)
                ? surfaces[i - 1].material.getIor(wavelengthNm) : 1.0f;
            const auto refracted = refract(*intersection, tracedRay, etaPrevious, etaSurface);
            if (!refracted)
                return false;
            tracedRay = {refracted->startPoint, refracted->direction.invert()};
            hits[hitIndex].outgoing = glm::normalize(glm::vec2(-tracedRay.direction.z, tracedRay.direction.y));
        }

        constexpr float longDistance = 1.0e5f;
        const ross::Vector3f exitPoint = tracedRay.startPoint + tracedRay.direction * longDistance;
        points.push_back({-exitPoint.z, exitPoint.y});
        return true;
    }

};

RealisticCamera* findActiveRealisticCamera(Scene& scene)
{
    CameraInstance* instance = scene.getActiveCamera();
    if (instance == nullptr || instance->getCamera() == nullptr)
        return nullptr;
    return instance->getCamera()->CastOrNullptr<RealisticCamera>();
}

}

LensViewerPanel::LensViewerPanel(std::string name, Scene& scene)
    : ImGuiComponent(std::move(name)), scene(scene) {}

void LensViewerPanel::retraceRays(const RealisticCamera& camera)
{
    rays.clear();
    settingsDirty = false;
    fitPending = true;
    if (camera.rossLens == nullptr || camera.rossLens->surfaces.empty())
        return;

    const ross::CameraLens& lens = *camera.rossLens;
    const LensViewerRaytracer tracer(lens);

    const uint32_t resolutionHeight = std::max(1u, camera.sensor.resolutionHeight);
    const uint32_t stride = std::max(1u, static_cast<uint32_t>(std::max(1, pixelStride)));
    std::vector<uint32_t> rows;
    for (uint32_t y = 0; y < resolutionHeight; y += stride)
        rows.push_back(y);
    if (rows.back() != resolutionHeight - 1)
        rows.push_back(resolutionHeight - 1);

    struct SpectralRay { float wavelength; uint32_t color; };
    const SpectralRay spectralRays[] = {
        {ross::FraunhoferLines::C, IM_COL32(255, 75, 75, 210)},
        {ross::FraunhoferLines::e, IM_COL32(90, 255, 110, 210)},
        {ross::FraunhoferLines::F, IM_COL32(90, 140, 255, 210)},
    };
    for (const uint32_t y : rows) {
        const float ny = 1.0f - (static_cast<float>(y) + 0.5f) / static_cast<float>(resolutionHeight) * 2.0f;
        ross::Ray filmRay;
        if (!camera.makeFilmRay(filmRay, 0.0f, ny, ross::Vector2f(0.5f, 0.5f)))
            continue;
        for (const auto& spectral : spectralRays) {
            TracedRay tracedRay;
            tracedRay.color = spectral.color;
            tracedRay.vignetted = !tracer.trace(filmRay, spectral.wavelength, tracedRay.points, tracedRay.hits);
            if (tracedRay.points.size() >= 2)
                rays.push_back(std::move(tracedRay));
        }
    }
}

void LensViewerPanel::drawCanvas(const RealisticCamera* camera)
{
    // Gather the plotted bounds (surfaces + traced rays) in the lens' native z/height frame.
    float zMin = 0.0f, zMax = 0.0f, heightMax = 0.0f;
    if (camera != nullptr && camera->rossLens != nullptr) {
        for (const auto& surf : camera->rossLens->surfaces) {
            const float z = surf.center;
            zMin = std::min(zMin, z);
            zMax = std::max(zMax, z);
            heightMax = std::max(heightMax, surf.apertureRadius);
        }
    }
    for (const auto& ray : rays) {

        const size_t pointCount = ray.vignetted ? ray.points.size() : ray.points.size() - 1;
        for (size_t i = 0; i < pointCount; ++i) {
            zMin = std::min(zMin, ray.points[i].x);
            zMax = std::max(zMax, ray.points[i].x);
            heightMax = std::max(heightMax, std::abs(ray.points[i].y));
        }
    }
    zMax = std::max(zMax, zMin + 1e-3f);
    heightMax = std::max(heightMax, 1e-3f);

    const ImVec2 canvasSize(ImGui::GetContentRegionAvail().x, std::max(200.0f, ImGui::GetContentRegionAvail().y));
    const ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##LensViewerCanvas", canvasSize);
    const bool hovered = ImGui::IsItemHovered();

    if (fitPending && canvasSize.x > 1.0f) {
        fitPending = false;
        const float zRange = (zMax - zMin) * 1.15f;
        const float hRange = heightMax * 2.0f * 1.4f;
        zoom = std::max(1e-4f, std::min(canvasSize.x / zRange, canvasSize.y / hRange));
        pan = glm::vec2((zMin + zMax) * 0.5f, 0.0f);
    }
    if (hovered) {
        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            pan -= glm::vec2(ImGui::GetIO().MouseDelta.x, -ImGui::GetIO().MouseDelta.y) / zoom;
        const float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f)
            zoom *= std::pow(1.1f, wheel);
    }

    const ImVec2 origin(canvasPos.x + canvasSize.x * 0.5f, canvasPos.y + canvasSize.y * 0.5f);
    const auto toScreen = [&](glm::vec2 p) {
        return ImVec2(
            origin.x + (p.x - pan.x) * zoom,
            origin.y - (p.y - pan.y) * zoom);
    };

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->PushClipRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), true);
    drawList->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(24, 24, 26, 255));

    // Optical axis.
    drawList->AddLine(toScreen({zMin, 0.0f}), toScreen({zMax, 0.0f}), IM_COL32(80, 80, 84, 255));

    if (camera != nullptr && camera->sensorHeightCm > 0.0f) {
        drawList->AddLine(toScreen({0.0f, -camera->sensorHeightCm * 0.5f}), toScreen({0.0f, camera->sensorHeightCm * 0.5f}),
            IM_COL32(200, 200, 60, 255), 2.0f);
    }

    if (camera != nullptr && camera->rossLens != nullptr) {
        for (const auto& surf : camera->rossLens->surfaces) {
            const float z = surf.center;
            if (surf.isAperture()) {
                // Draw iris blades: two short bars radiating outward from the stop radius.
                const float r0 = surf.apertureRadius;
                const float r1 = r0 + std::max(heightMax * 0.08f, 1e-3f);
                drawList->AddLine(toScreen({z, r0}), toScreen({z, r1}), IM_COL32(220, 220, 220, 255), 2.0f);
                drawList->AddLine(toScreen({z, -r0}), toScreen({z, -r1}), IM_COL32(220, 220, 220, 255), 2.0f);
                continue;
            }

            constexpr int arcSegments = 64;
            const ImU32 glassColor = IM_COL32(120, 190, 255, 220);
            ImVec2 prev{};
            bool havePrev = false;
            for (int s = 0; s <= arcSegments; ++s) {
                const float h = -surf.apertureRadius + surf.apertureRadius * 2.0f * static_cast<float>(s) / arcSegments;
                float surfaceZ = z;
                switch (surf.geometry) {
                    case ross::LensGeometry::SPHERICAL:
                    case ross::LensGeometry::CYLINDER_X: {
                        // Cylinder X has the same circular profile in this y/z section.
                        const float radius = surf.curvatureRadius;
                        const float under = radius * radius - h * h;
                        if (radius == 0.0f || under < 0.0f) {
                            havePrev = false;
                            continue;
                        }
                        surfaceZ = z - radius + std::copysign(std::sqrt(under), radius);
                        break;
                    }
                    case ross::LensGeometry::ASPHERICAL: {
                        if (!surf.asphericCoefficientsIndex.has_value()) {
                            havePrev = false;
                            continue;
                        }
                        const int index = *surf.asphericCoefficientsIndex;
                        if (index < 0 || static_cast<size_t>(index) >= camera->rossLens->asphericCoefficients.size()) {
                            havePrev = false;
                            continue;
                        }
                        // libross defines sag in its internal -z frame; native/display z therefore
                        // subtracts it from the vertex position.
                        surfaceZ = z - ross::evaluateAsphericalSurface(
                            std::abs(h), surf.curvatureRadius,
                            camera->rossLens->asphericCoefficients[index]);
                        break;
                    }
                    case ross::LensGeometry::CYLINDER_Y:
                    case ross::LensGeometry::PLANAR:
                        // Cylinder Y curves along x and is planar in this y/z section.
                        break;
                }
                const ImVec2 sp = toScreen({surfaceZ, h});
                if (havePrev)
                    drawList->AddLine(prev, sp, glassColor, 2.0f);
                prev = sp;
                havePrev = true;
            }
        }
    }

    const float normalLength = std::max(heightMax * 0.05f, 1e-3f);
    const float directionLength = normalLength * 1.5f;
    for (const auto& ray : rays) {
        // Keep the wavelength color even when a path is rejected. Otherwise every failed ray
        // becomes red and can be mistaken for the Fraunhofer C line.
        const ImU32 color = ray.color;
        for (size_t i = 1; i < ray.points.size(); ++i)
            drawList->AddLine(toScreen(ray.points[i - 1]), toScreen(ray.points[i]), color, 1.0f);
        for (const auto& hit : ray.hits) {
            const ImVec2 hitScreen = toScreen(hit.position);
            drawList->AddCircleFilled(hitScreen, 2.5f, IM_COL32(255, 255, 255, 240));
            drawList->AddLine(hitScreen, toScreen(hit.position + hit.normal * normalLength),
                IM_COL32(120, 255, 140, 200), 1.0f);
            if (glm::dot(hit.outgoing, hit.outgoing) > 0.0f)
                drawList->AddLine(hitScreen, toScreen(hit.position + hit.outgoing * directionLength), color, 2.0f);
        }
    }

    drawList->PopClipRect();
}

void LensViewerPanel::renderUi()
{
    const bool visible = ImGui::Begin(
        name.c_str(), nullptr, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    if (!visible) {
        ImGui::End();
        return;
    }
    if (ImGui::IsWindowAppearing())
        fitPending = true;

    RealisticCamera* camera = findActiveRealisticCamera(scene);
    if (camera == nullptr) {
        ImGui::TextDisabled("No realistic camera active.");
        ImGui::End();
        return;
    }

    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::DragInt("Pixel stride", &pixelStride, 1.0f, 1, static_cast<int>(camera->sensor.resolutionHeight)))
        settingsDirty = true;
    ImGui::SameLine();
    if (ImGui::Button("Fit View"))
        fitPending = true;

    const bool opticsDirty = camera->consumeOpticsDirty();
    if (settingsDirty || opticsDirty)
        retraceRays(*camera);

    drawCanvas(camera);

    ImGui::End();
}
