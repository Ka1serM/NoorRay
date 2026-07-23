#include "UI/MathInput.h"

#include <array>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>

#include <imgui_internal.h>

namespace {

class ExpressionParser {
public:
    ExpressionParser(const char* expression, const long double initialValue)
        : cursor(expression), initialValue(initialValue)
    {
    }

    bool parse(long double& result)
    {
        skipWhitespace();
        if (*cursor == '\0')
            return false;
        result = parseExpression(true);
        skipWhitespace();
        return valid && *cursor == '\0' && std::isfinite(result);
    }

private:
    long double parseExpression(const bool allowRelativePrefix)
    {
        skipWhitespace();
        long double result;
        if (allowRelativePrefix && *cursor == '+')
            result = initialValue;
        else
            result = parseTerm(allowRelativePrefix);

        while (valid)
        {
            skipWhitespace();
            const char operation = *cursor;
            if (operation != '+' && operation != '-')
                break;
            ++cursor;
            const long double operand = parseTerm(false);
            result = operation == '+' ? result + operand : result - operand;
        }
        return result;
    }

    long double parseTerm(const bool allowRelativePrefix)
    {
        skipWhitespace();
        long double result;
        if (allowRelativePrefix && (*cursor == '*' || *cursor == '/'))
            result = initialValue;
        else
            result = parseUnary();

        while (valid)
        {
            skipWhitespace();
            const char operation = *cursor;
            if (operation != '*' && operation != '/')
                break;
            ++cursor;
            const long double operand = parseUnary();
            if (operation == '/' && operand == 0.0L)
            {
                valid = false;
                return result;
            }
            result = operation == '*' ? result * operand : result / operand;
        }
        return result;
    }

    long double parseUnary()
    {
        skipWhitespace();
        if (*cursor == '+')
        {
            ++cursor;
            return parseUnary();
        }
        if (*cursor == '-')
        {
            ++cursor;
            return -parseUnary();
        }
        return parsePrimary();
    }

    long double parsePrimary()
    {
        skipWhitespace();
        if (*cursor == '(')
        {
            ++cursor;
            const long double result = parseExpression(false);
            skipWhitespace();
            if (*cursor != ')')
            {
                valid = false;
                return 0.0L;
            }
            ++cursor;
            return result;
        }

        char* numberEnd = nullptr;
        const long double result = std::strtold(cursor, &numberEnd);
        if (numberEnd == cursor)
        {
            valid = false;
            return 0.0L;
        }
        cursor = numberEnd;
        return result;
    }

    void skipWhitespace()
    {
        while (ImCharIsBlankA(*cursor))
            ++cursor;
    }

    const char* cursor;
    long double initialValue;
    bool valid = true;
};

template<typename T>
void storeExpressionResult(
    const long double result, T* outputValue,
    const long double typeMin, const long double typeMax)
{
    if (result <= typeMin)
        *outputValue = std::numeric_limits<T>::lowest();
    else if (result >= typeMax)
        *outputValue = std::numeric_limits<T>::max();
    else
        *outputValue = static_cast<T>(result);
}

struct EditState {
    ImGuiDataType dataType = ImGuiDataType_COUNT;
    ImGuiDataTypeStorage initialValue{};
    std::string expression;
};

std::unordered_map<ImGuiID, EditState> editStates;

bool processScalar(
    const ImGuiID id, const ImGuiDataType dataType, void* value,
    const void* min, const void* max, const ImGuiSliderFlags sliderFlags,
    const char* format)
{
    ImGuiContext& context = *GImGui;
    ImGuiInputTextState& inputState = context.InputTextState;
    const bool active = context.ActiveId == id && inputState.ID == id;
    const bool deactivated =
        context.DeactivatedItemData.ID == id
        && context.DeactivatedItemData.ElapseFrame >= context.FrameCount;

    if (active || deactivated)
    {
        auto [iterator, inserted] = editStates.try_emplace(id);
        EditState& editState = iterator->second;
        if (inserted || editState.dataType != dataType)
        {
            editState.dataType = dataType;
            const char* initialText = inputState.ID == id
                ? inputState.TextToRevertTo.Data : nullptr;
            const char* scanFormat = format != nullptr
                ? format : ImGui::DataTypeGetInfo(dataType)->ScanFmt;
            if (initialText != nullptr)
            {
                ImGui::DataTypeApplyFromText(
                    initialText, dataType, &editState.initialValue,
                    scanFormat, nullptr);
            }
            else
            {
                std::memcpy(
                    &editState.initialValue, value,
                    ImGui::DataTypeGetInfo(dataType)->Size);
            }
        }

        if (inputState.ID == id && inputState.TextA.Data != nullptr)
            editState.expression.assign(
                inputState.TextA.Data,
                static_cast<std::size_t>(inputState.TextLen));
    }

    if (!deactivated)
        return false;

    const auto iterator = editStates.find(id);
    if (iterator == editStates.end())
        return false;

    const EditState editState = iterator->second;
    editStates.erase(iterator);
    const std::size_t dataSize = ImGui::DataTypeGetInfo(dataType)->Size;
    const bool valid = MathInput::applyExpression(
        editState.expression.c_str(), dataType,
        &editState.initialValue, value);
    if (!valid)
    {
        std::memcpy(value, &editState.initialValue, dataSize);
        return false;
    }

    if ((sliderFlags & ImGuiSliderFlags_ClampOnInput) != 0)
        ImGui::DataTypeClamp(dataType, value, min, max);
    const bool changed =
        std::memcmp(value, &editState.initialValue, dataSize) != 0;
    if (!changed)
        return false;
    ImGui::MarkItemEdited(id);
    return true;
}

std::array<ImGuiID, 3> float3ComponentIds(const char* label)
{
    std::array<ImGuiID, 3> ids{};
    ImGui::PushID(label);
    for (int index = 0; index < 3; ++index)
    {
        ImGui::PushID(index);
        ids[index] = ImGui::GetCurrentWindow()->GetID("");
        ImGui::PopID();
    }
    ImGui::PopID();
    return ids;
}

}

namespace MathInput {

bool applyExpression(
    const char* expression, const ImGuiDataType dataType,
    const void* initialValue, void* outputValue)
{
    long double initial;
    switch (dataType)
    {
    case ImGuiDataType_S8:     initial = *static_cast<const ImS8*>(initialValue); break;
    case ImGuiDataType_U8:     initial = *static_cast<const ImU8*>(initialValue); break;
    case ImGuiDataType_S16:    initial = *static_cast<const ImS16*>(initialValue); break;
    case ImGuiDataType_U16:    initial = *static_cast<const ImU16*>(initialValue); break;
    case ImGuiDataType_S32:    initial = *static_cast<const ImS32*>(initialValue); break;
    case ImGuiDataType_U32:    initial = *static_cast<const ImU32*>(initialValue); break;
    case ImGuiDataType_S64:    initial = *static_cast<const ImS64*>(initialValue); break;
    case ImGuiDataType_U64:    initial = *static_cast<const ImU64*>(initialValue); break;
    case ImGuiDataType_Float:  initial = *static_cast<const float*>(initialValue); break;
    case ImGuiDataType_Double: initial = *static_cast<const double*>(initialValue); break;
    case ImGuiDataType_COUNT:  return false;
    }

    long double result = 0.0L;
    ExpressionParser parser(expression, initial);
    if (!parser.parse(result))
        return false;

    switch (dataType)
    {
    case ImGuiDataType_S8:
        storeExpressionResult(
            result,
            static_cast<ImS8*>(outputValue),
            std::numeric_limits<ImS8>::lowest(),
            std::numeric_limits<ImS8>::max());
        break;
    case ImGuiDataType_U8:
        storeExpressionResult(
            result,
            static_cast<ImU8*>(outputValue),
            std::numeric_limits<ImU8>::lowest(),
            std::numeric_limits<ImU8>::max());
        break;
    case ImGuiDataType_S16:
        storeExpressionResult(
            result,
            static_cast<ImS16*>(outputValue),
            std::numeric_limits<ImS16>::lowest(),
            std::numeric_limits<ImS16>::max());
        break;
    case ImGuiDataType_U16:
        storeExpressionResult(
            result,
            static_cast<ImU16*>(outputValue),
            std::numeric_limits<ImU16>::lowest(),
            std::numeric_limits<ImU16>::max());
        break;
    case ImGuiDataType_S32:
        storeExpressionResult(
            result,
            static_cast<ImS32*>(outputValue),
            std::numeric_limits<ImS32>::lowest(),
            std::numeric_limits<ImS32>::max());
        break;
    case ImGuiDataType_U32:
        storeExpressionResult(
            result,
            static_cast<ImU32*>(outputValue),
            std::numeric_limits<ImU32>::lowest(),
            std::numeric_limits<ImU32>::max());
        break;
    case ImGuiDataType_S64:
        storeExpressionResult(
            result,
            static_cast<ImS64*>(outputValue),
            static_cast<long double>(std::numeric_limits<ImS64>::lowest()),
            static_cast<long double>(std::numeric_limits<ImS64>::max()));
        break;
    case ImGuiDataType_U64:
        storeExpressionResult(
            result,
            static_cast<ImU64*>(outputValue),
            static_cast<long double>(std::numeric_limits<ImU64>::lowest()),
            static_cast<long double>(std::numeric_limits<ImU64>::max()));
        break;
    case ImGuiDataType_Float:
        storeExpressionResult(
            result,
            static_cast<float*>(outputValue), -FLT_MAX, FLT_MAX);
        break;
    case ImGuiDataType_Double:
        storeExpressionResult(
            result,
            static_cast<double*>(outputValue), -DBL_MAX, DBL_MAX);
        break;
    case ImGuiDataType_COUNT:
        return false;
    }
    return true;
}

bool DragFloat(
    const char* label, float* value, const float speed,
    const float min, const float max, const char* format,
    const ImGuiSliderFlags flags)
{
    const ImGuiID id = ImGui::GetCurrentWindow()->GetID(label);
    const bool changed =
        ImGui::DragFloat(label, value, speed, min, max, format, flags);
    return processScalar(
        id, ImGuiDataType_Float, value, &min, &max, flags, format) || changed;
}

bool DragFloat3(
    const char* label, float values[3], const float speed,
    const float min, const float max, const char* format,
    const ImGuiSliderFlags flags)
{
    const std::array<ImGuiID, 3> ids = float3ComponentIds(label);
    bool changed =
        ImGui::DragFloat3(label, values, speed, min, max, format, flags);
    for (int index = 0; index < 3; ++index)
        changed |= processScalar(
            ids[index], ImGuiDataType_Float, &values[index],
            &min, &max, flags, format);
    return changed;
}

bool DragInt(
    const char* label, int* value, const float speed,
    const int min, const int max, const char* format,
    const ImGuiSliderFlags flags)
{
    const ImGuiID id = ImGui::GetCurrentWindow()->GetID(label);
    const bool changed =
        ImGui::DragInt(label, value, speed, min, max, format, flags);
    return processScalar(
        id, ImGuiDataType_S32, value, &min, &max, flags, format) || changed;
}

bool SliderFloat(
    const char* label, float* value, const float min, const float max,
    const char* format, const ImGuiSliderFlags flags)
{
    const ImGuiID id = ImGui::GetCurrentWindow()->GetID(label);
    const bool changed =
        ImGui::SliderFloat(label, value, min, max, format, flags);
    return processScalar(
        id, ImGuiDataType_Float, value, &min, &max, flags, format) || changed;
}

bool SliderInt(
    const char* label, int* value, const int min, const int max,
    const char* format, const ImGuiSliderFlags flags)
{
    const ImGuiID id = ImGui::GetCurrentWindow()->GetID(label);
    const bool changed =
        ImGui::SliderInt(label, value, min, max, format, flags);
    return processScalar(
        id, ImGuiDataType_S32, value, &min, &max, flags, format) || changed;
}

bool InputInt(
    const char* label, int* value, const int step, const int stepFast,
    ImGuiInputTextFlags flags)
{
    flags &= ~(ImGuiInputTextFlags_CharsDecimal
        | ImGuiInputTextFlags_CharsScientific
        | ImGuiInputTextFlags_CharsHexadecimal);
    const ImGuiID id = ImGui::GetCurrentWindow()->GetID(label);
    const bool changed =
        ImGui::InputInt(label, value, step, stepFast, flags);
    return processScalar(
        id, ImGuiDataType_S32, value, nullptr, nullptr, 0, "%d") || changed;
}

bool InputScalar(
    const char* label, const ImGuiDataType dataType, void* value,
    const void* step, const void* stepFast, const char* format,
    ImGuiInputTextFlags flags)
{
    flags &= ~(ImGuiInputTextFlags_CharsDecimal
        | ImGuiInputTextFlags_CharsScientific
        | ImGuiInputTextFlags_CharsHexadecimal);
    const ImGuiID id = ImGui::GetCurrentWindow()->GetID(label);
    const bool changed =
        ImGui::InputScalar(
            label, dataType, value, step, stepFast, format, flags);
    return processScalar(
        id, dataType, value, nullptr, nullptr, 0, format) || changed;
}

}
