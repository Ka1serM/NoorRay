#pragma once

#include "UI/ImGuiComponent.h"
#include <glm/vec2.hpp>
#include <cstdint>
#include <vector>

class Scene;
class RealisticCamera;

class LensViewerPanel : public ImGuiComponent {
public:
    LensViewerPanel(std::string name, Scene& scene);
    void renderUi() override;

    struct SurfaceHit {
        glm::vec2 position;  // (z, height), in the lens' native units
        glm::vec2 normal;    // libross intersection normal in the same (z, height) frame
        glm::vec2 outgoing;  // post-refraction ray direction; zero when refraction failed
    };

private:
    struct TracedRay {
        std::vector<glm::vec2> points; // (z, height), in the lens' native units
        std::vector<SurfaceHit> hits;
        bool vignetted = false;
        uint32_t color = 0;
    };

    Scene& scene;
    std::vector<TracedRay> rays;
    bool settingsDirty = true;
    bool fitPending = true;
    int pixelStride = 100; // trace a ray fan for every Nth sensor row (physical pixel grid)
    glm::vec2 pan{0.0f, 0.0f};
    float zoom = 1.0f;

    void retraceRays(const RealisticCamera& camera);
    void drawCanvas(const RealisticCamera* camera);
};
