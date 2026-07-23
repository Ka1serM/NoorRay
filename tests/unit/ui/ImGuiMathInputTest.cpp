#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "UI/MathInput.h"

namespace {

struct ImGuiTestContext {
    ImGuiTestContext()
    {
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2(640.0f, 480.0f);
        io.DeltaTime = 1.0f / 60.0f;
        io.ConfigDragClickToInputText = true;
        unsigned char* pixels = nullptr;
        int width = 0;
        int height = 0;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    }

    ~ImGuiTestContext()
    {
        ImGui::DestroyContext();
    }
};

}

TEST_CASE("ImGui scalar inputs evaluate arithmetic expressions", "[ui][imgui]") {
    SECTION("floating-point multiplication and division") {
        float value = 12.0f;
        const float initialValue = value;
        REQUIRE(MathInput::applyExpression(
            "* 2", ImGuiDataType_Float, &initialValue, &value));
        REQUIRE(value == Catch::Approx(24.0f));

        const float multipliedValue = value;
        REQUIRE(MathInput::applyExpression(
            "/ 4", ImGuiDataType_Float, &multipliedValue, &value));
        REQUIRE(value == Catch::Approx(6.0f));
    }

    SECTION("addition supports negative operands") {
        int value = 10;
        const int initialValue = value;
        REQUIRE(MathInput::applyExpression(
            "+-3", ImGuiDataType_S32, &initialValue, &value));
        REQUIRE(value == 7);
    }

    SECTION("absolute expressions support precedence and parentheses") {
        int value = 10;
        const int initialValue = value;
        REQUIRE(MathInput::applyExpression(
            "10 + 2 * 3", ImGuiDataType_S32, &initialValue, &value));
        REQUIRE(value == 16);

        REQUIRE(MathInput::applyExpression(
            "(10 + 2) * 3", ImGuiDataType_S32, &initialValue, &value));
        REQUIRE(value == 36);

        REQUIRE(MathInput::applyExpression(
            "-2", ImGuiDataType_S32, &initialValue, &value));
        REQUIRE(value == -2);
    }

    SECTION("integer multiplication accepts a fractional operand") {
        int value = 9;
        const int initialValue = value;
        REQUIRE(MathInput::applyExpression(
            "* 0.5", ImGuiDataType_S32, &initialValue, &value));
        REQUIRE(value == 4);
    }

    SECTION("division by zero is ignored") {
        double value = 8.0;
        const double initialValue = value;
        REQUIRE_FALSE(MathInput::applyExpression(
            "/ 0", ImGuiDataType_Double, &initialValue, &value));
        REQUIRE(value == Catch::Approx(8.0));
    }
}

TEST_CASE("MathInput commits expressions through an ImGui input widget", "[ui][imgui]") {
    ImGuiTestContext context;
    int value = 10;
    ImVec2 inputCenter;

    const auto drawFrame = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(400.0f, 200.0f));
        ImGui::Begin("Test");
        ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, false);
        MathInput::InputInt(
            "Value", &value, 0, 0, ImGuiInputTextFlags_CharsDecimal);
        ImGui::PopItemFlag();
        const ImVec2 inputMin = ImGui::GetItemRectMin();
        const ImVec2 inputMax = ImGui::GetItemRectMax();
        inputCenter = ImVec2(
            (inputMin.x + inputMax.x) * 0.5f,
            (inputMin.y + inputMax.y) * 0.5f);
        ImGui::End();
        ImGui::EndFrame();
    };

    drawFrame();

    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(inputCenter.x, inputCenter.y);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    drawFrame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    drawFrame();

    io.AddInputCharactersUTF8("(10 + 2) * 3");
    drawFrame();
    io.AddKeyEvent(ImGuiKey_Enter, true);
    drawFrame();
    io.AddKeyEvent(ImGuiKey_Enter, false);
    drawFrame();

    REQUIRE(value == 36);
}

TEST_CASE("MathInput commits expressions through an ImGui drag widget", "[ui][imgui]") {
    ImGuiTestContext context;
    float value = 10.0f;
    ImVec2 inputCenter;

    const auto drawFrame = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(400.0f, 200.0f));
        ImGui::Begin("Test");
        ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, false);
        MathInput::DragFloat("Value", &value);
        ImGui::PopItemFlag();
        const ImVec2 inputMin = ImGui::GetItemRectMin();
        const ImVec2 inputMax = ImGui::GetItemRectMax();
        inputCenter = ImVec2(
            (inputMin.x + inputMax.x) * 0.5f,
            (inputMin.y + inputMax.y) * 0.5f);
        ImGui::End();
        ImGui::EndFrame();
    };

    drawFrame();

    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(inputCenter.x, inputCenter.y);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    drawFrame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    drawFrame();

    io.AddInputCharactersUTF8("10 * 2");
    drawFrame();
    io.AddKeyEvent(ImGuiKey_Enter, true);
    drawFrame();
    io.AddKeyEvent(ImGuiKey_Enter, false);
    drawFrame();

    REQUIRE(value == Catch::Approx(20.0f));
}

TEST_CASE("MathInput commits expressions through a vector component", "[ui][imgui]") {
    ImGuiTestContext context;
    float values[3]{10.0f, 20.0f, 30.0f};
    ImVec2 firstComponentCenter;

    const auto drawFrame = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(400.0f, 200.0f));
        ImGui::Begin("Test");
        ImGui::SetNextItemWidth(300.0f);
        ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, false);
        MathInput::DragFloat3("##Value", values);
        ImGui::PopItemFlag();
        const ImVec2 inputMin = ImGui::GetItemRectMin();
        const ImVec2 inputMax = ImGui::GetItemRectMax();
        firstComponentCenter = ImVec2(
            inputMin.x + (inputMax.x - inputMin.x) / 6.0f,
            (inputMin.y + inputMax.y) * 0.5f);
        ImGui::End();
        ImGui::EndFrame();
    };

    drawFrame();

    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(firstComponentCenter.x, firstComponentCenter.y);
    io.AddKeyEvent(ImGuiKey_LeftCtrl, true);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    drawFrame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    io.AddKeyEvent(ImGuiKey_LeftCtrl, false);
    io.AddKeyEvent(ImGuiMod_Ctrl, false);
    drawFrame();

    io.AddInputCharactersUTF8("* 2");
    drawFrame();
    io.AddKeyEvent(ImGuiKey_Enter, true);
    drawFrame();
    io.AddKeyEvent(ImGuiKey_Enter, false);
    drawFrame();

    REQUIRE(values[0] == Catch::Approx(20.0f));
    REQUIRE(values[1] == Catch::Approx(20.0f));
    REQUIRE(values[2] == Catch::Approx(30.0f));
}

TEST_CASE("MathInput commits an expression when focus moves away", "[ui][imgui]") {
    ImGuiTestContext context;
    float value = 10.0f;
    ImVec2 inputCenter;
    ImVec2 buttonCenter;

    const auto drawFrame = [&] {
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0.0f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(400.0f, 200.0f));
        ImGui::Begin("Test");
        ImGui::PushItemFlag(ImGuiItemFlags_LiveEditOnInputScalar, false);
        MathInput::DragFloat("Value", &value);
        ImGui::PopItemFlag();
        ImVec2 itemMin = ImGui::GetItemRectMin();
        ImVec2 itemMax = ImGui::GetItemRectMax();
        inputCenter = ImVec2(
            (itemMin.x + itemMax.x) * 0.5f,
            (itemMin.y + itemMax.y) * 0.5f);
        ImGui::Button("Other");
        itemMin = ImGui::GetItemRectMin();
        itemMax = ImGui::GetItemRectMax();
        buttonCenter = ImVec2(
            (itemMin.x + itemMax.x) * 0.5f,
            (itemMin.y + itemMax.y) * 0.5f);
        ImGui::End();
        ImGui::EndFrame();
    };

    drawFrame();

    ImGuiIO& io = ImGui::GetIO();
    io.AddMousePosEvent(inputCenter.x, inputCenter.y);
    io.AddKeyEvent(ImGuiKey_LeftCtrl, true);
    io.AddKeyEvent(ImGuiMod_Ctrl, true);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    drawFrame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    io.AddKeyEvent(ImGuiKey_LeftCtrl, false);
    io.AddKeyEvent(ImGuiMod_Ctrl, false);
    drawFrame();
    io.AddInputCharactersUTF8("* 2");
    drawFrame();

    io.AddMousePosEvent(buttonCenter.x, buttonCenter.y);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    drawFrame();
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    drawFrame();
    drawFrame();

    REQUIRE(value == Catch::Approx(20.0f));
}
