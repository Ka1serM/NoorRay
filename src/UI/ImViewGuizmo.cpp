#include "ImViewGuizmo.h"
#include <algorithm>

bool ImViewGuizmo::Manipulate(vec3& position, quat& rotation, ImVec2 pivotPos, float size, float snapDistance, float mouseSpeed)
{
    auto& style = GetStyle();
    auto& ctx = GetContext();
    bool wasModified = false;
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // Generate Matrices & Axis Data
    mat4 worldMatrix = translate(mat4(1.0f), position) * mat4_cast(rotation);
    mat4 viewMatrix = inverse(worldMatrix);

    mat4 gizmoViewMatrix = mat4(mat3(viewMatrix));
    constexpr float orthoSize = 1.2f;
    mat4 gizmoProjectionMatrix = ortho(-orthoSize, orthoSize, orthoSize, -orthoSize, -100.0f, 100.0f);
    mat4 gizmoMvp = gizmoProjectionMatrix * gizmoViewMatrix;

    std::vector<GizmoAxis> axes;
    for (int i = 0; i < 3; ++i) {
        axes.push_back({i * 2, i, (gizmoViewMatrix * vec4(axisVectors[i], 0)).z, axisVectors[i]});
        axes.push_back({i * 2 + 1, i, (gizmoViewMatrix * vec4(-axisVectors[i], 0)).z, -axisVectors[i]});
    }

    std::ranges::sort(axes, [](const GizmoAxis& a, const GizmoAxis& b) {
        return a.depth < b.depth;
    });

    auto worldToScreen = [&](const vec3& worldPos) -> ImVec2 {
        const vec4 clipPos = gizmoMvp * vec4(worldPos, 1.0f);
        if (clipPos.w == 0.0f) return {-1, -1};
        const vec3 ndc = vec3(clipPos) / clipPos.w;
        float x = pivotPos.x - (ndc.x * 0.5f + 0.5f) * size;
        float y = pivotPos.y + (ndc.y * 0.5f + 0.5f) * size;
        return {x, y};
    };

    // 2D Selection Logic
    ctx.hoveredAxisID = -1;
    ImVec2 rectMin = ImVec2(pivotPos.x - size, pivotPos.y);
    ImVec2 rectMax = ImVec2(pivotPos.x, pivotPos.y + size);
    if (ImGui::IsMouseHoveringRect(rectMin, rectMax)) {
        ImVec2 mousePos = io.MousePos;
        float minDistanceSq = style.circleRadius * style.circleRadius;

        for (const auto& axis : axes) {
            if (axis.depth < -0.1f)
                continue;
            ImVec2 handlePos = worldToScreen(axis.direction * style.lineLength);
            float dx = handlePos.x - mousePos.x;
            float dy = handlePos.y - mousePos.y;
            float distSq = dx*dx + dy*dy;
            if (distSq < minDistanceSq) {
                minDistanceSq = distSq;
                ctx.hoveredAxisID = axis.id;
            }
        }

        if (ctx.hoveredAxisID == -1) {
            ImVec2 centerPos = worldToScreen(origin);
            float dx = centerPos.x - mousePos.x;
            float dy = centerPos.y - mousePos.y;
            if ((dx*dx + dy*dy) < style.bigCircleRadius * style.bigCircleRadius)
                ctx.hoveredAxisID = 6;
        }
    }

    // Draw Geometry
    if (ctx.hoveredAxisID == 6)
        drawList->AddCircleFilled(worldToScreen(origin), style.bigCircleRadius, IM_COL32(255, 255, 255, 50));

    for (const auto& [id, axis_index, depth, direction] : axes) {
        float factor = mix(style.fadeFactor, 1.0f, (depth + 1.0f) * 0.5f);
        ImColor axis_color_im = style.axisColors[axis_index];
        float h, s, v;
        ImGui::ColorConvertRGBtoHSV(axis_color_im.Value.x, axis_color_im.Value.y, axis_color_im.Value.z, h, s, v);
        ImU32 final_color = ImColor::HSV(h, s, v * factor);

        ImVec2 handlePos = worldToScreen(direction * style.lineLength);
        drawList->AddLine(worldToScreen(origin), handlePos, final_color, style.lineWidth);
        drawList->AddCircleFilled(handlePos, style.circleRadius, final_color);
        if (ctx.hoveredAxisID == id)
            drawList->AddCircle(handlePos, style.circleRadius + 2.0f, style.highlightColor, 0, style.highlightWidth);
    }

    // Draw Text Overlay
    for (const auto& axis : axes) {
        if (axis.depth < -0.1f)
            continue;
        
        ImVec2 textPos = worldToScreen(axis.direction * style.lineLength);
        const char* label = style.axisLabels[axis.axisIndex];
        ImVec2 textSize = ImGui::CalcTextSize(label);
        drawList->AddText(ImGui::GetFont(), ImGui::GetFontSize(),
            {textPos.x - textSize.x * 0.5f, textPos.y - textSize.y * 0.5f}, IM_COL32(0, 0, 0, 255), label);
    }

    // Drag logic - only start when hovering gizmo
    if (ImGui::IsMouseDown(0)) {
        if (!ctx.isDragging && ctx.hoveredAxisID == 6)
            ctx.isDragging = true; // Start drag

        if (ctx.isDragging) {
            float yawAngle = -io.MouseDelta.x * mouseSpeed;
            float pitchAngle = -io.MouseDelta.y * mouseSpeed;

            quat yawRotation = angleAxis(yawAngle, worldUp);
            vec3 rightAxis = rotation * right;
            quat pitchRotation = angleAxis(pitchAngle, rightAxis);

            quat totalRotation = yawRotation * pitchRotation;
            position = totalRotation * position;
            rotation = totalRotation * rotation;

            wasModified = true;
        }
    } else
        ctx.isDragging = false; // end drag

    // Snap logic
    if (ImGui::IsMouseReleased(0) && ctx.hoveredAxisID >= 0 && ctx.hoveredAxisID <= 5) {
        int axisIndex = ctx.hoveredAxisID / 2;
        float sign = (ctx.hoveredAxisID % 2 == 0) ? 1.0f : -1.0f;

        vec3 forward = sign * axisVectors[axisIndex];
        
        rotation = quatLookAt(forward, -worldUp);
        position = forward * snapDistance;

        wasModified = true;
    }

    return wasModified;
}