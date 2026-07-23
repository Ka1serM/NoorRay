#pragma once

#include <imgui.h>

namespace MathInput {

bool applyExpression(
    const char* expression, ImGuiDataType dataType,
    const void* initialValue, void* outputValue);

bool DragFloat(
    const char* label, float* value, float speed = 1.0f,
    float min = 0.0f, float max = 0.0f, const char* format = "%.3f",
    ImGuiSliderFlags flags = 0);
bool DragFloat3(
    const char* label, float values[3], float speed = 1.0f,
    float min = 0.0f, float max = 0.0f, const char* format = "%.3f",
    ImGuiSliderFlags flags = 0);
bool DragInt(
    const char* label, int* value, float speed = 1.0f,
    int min = 0, int max = 0, const char* format = "%d",
    ImGuiSliderFlags flags = 0);

bool SliderFloat(
    const char* label, float* value, float min, float max,
    const char* format = "%.3f", ImGuiSliderFlags flags = 0);
bool SliderInt(
    const char* label, int* value, int min, int max,
    const char* format = "%d", ImGuiSliderFlags flags = 0);

bool InputInt(
    const char* label, int* value, int step = 1, int stepFast = 100,
    ImGuiInputTextFlags flags = 0);
bool InputScalar(
    const char* label, ImGuiDataType dataType, void* value,
    const void* step = nullptr, const void* stepFast = nullptr,
    const char* format = nullptr, ImGuiInputTextFlags flags = 0);

}
