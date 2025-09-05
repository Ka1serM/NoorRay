#pragma once
#define GLM_ENABLE_EXPERIMENTAL
#include "imgui.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace ImViewGuizmo {

    using namespace glm;

    struct Style {
        // Axis visuals
        float lineLength       = 0.6f;
        float lineWidth        = 4.0f;
        float circleRadius     = 14.0f;
        float bigCircleRadius  = 80.0f;
        float fadeFactor       = 0.5f;

        // Highlight
        ImU32 highlightColor   = IM_COL32(255, 255, 255, 255);
        float highlightWidth   = 2.0f;

        // Colors
        ImU32 axisColors[3] = {
            IM_COL32(230, 51, 51, 255),   // X
            IM_COL32(51, 230, 51, 255),   // Y
            IM_COL32(51, 128, 255, 255)   // Z
        };

        // Labels
        const char* axisLabels[3] = {"X", "Y", "Z"};
    };

    inline Style& GetStyle() {
        static Style style;
        return style;
    }

    // Gizmo Axis Struct
    struct GizmoAxis {
        int id;         // 0-5 for (+X,-X,+Y,-Y,+Z,-Z), 6=center
        int axisIndex;  // 0=X, 1=Y, 2=Z
        float depth;    // Screen-space depth
        vec3 direction; // 3D vector
    };

    static constexpr vec3 origin = {0.0f, 0.0f, 0.0f};
    static constexpr vec3 right = vec3(1, 0, 0);
    static constexpr vec3 worldUp = {0.0f, -1.0f, 0.0f};
    static constexpr vec3 axisVectors[3] = {{1,0,0}, {0,1,0}, {0,0,1}};

    struct Context {
        int hoveredAxisID = -1;
        bool isDragging = false;

        bool IsHovering() const { return hoveredAxisID != -1; }
        void Reset() { hoveredAxisID = -1; isDragging = false; }
    };

    // Global context
    inline Context& GetContext() {
        static Context ctx;
        return ctx;
    }

    inline bool IsUsing() {
        const Context& ctx = GetContext();
        return ctx.hoveredAxisID != -1 || ctx.isDragging;
    }

    inline bool IsHovering() {
        return GetContext().hoveredAxisID != -1;
    }

    bool Manipulate(vec3& position, quat& rotation, ImVec2 pivotPos, float size, float snapDistance =  5.f, float mouseSpeed = 0.005f);

}