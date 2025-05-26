#include "MainMenuBar.h"
#include <imgui.h>

MainMenuBar::MainMenuBar()
{
}

void MainMenuBar::setCallback(const std::string& action, std::function<void()> callback) {
    callbacks[action] = std::move(callback);
}

void MainMenuBar::renderUi() {
    if (ImGui::BeginMainMenuBar()) {
        renderFileMenu();
        ImGui::EndMainMenuBar();
    }
}

void MainMenuBar::renderFileMenu() {
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New"))
            if (callbacks.contains("File.New"))
                callbacks["File.New"]();
        
        if (ImGui::MenuItem("Open...", "Ctrl+O"))
            if (callbacks.contains("File.Open"))
                callbacks["File.Open"]();
        
        if (ImGui::BeginMenu("Open Recent")) {
            // Add recent files logic here
            ImGui::MenuItem("RecentFile1.obj");
            ImGui::MenuItem("RecentFile2.obj");
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Save", "Ctrl+S"))
            if (callbacks.contains("File.Save"))
                callbacks["File.Save"]();
        
        if (ImGui::MenuItem("Save As...", "Shift+Ctrl+S"))
            if (callbacks.contains("File.SaveAs"))
                callbacks["File.SaveAs"]();
        
        if (ImGui::MenuItem("Save Copy..."))
            if (callbacks.contains("File.SaveCopy"))
                callbacks["File.SaveCopy"]();

        ImGui::Separator();

        if (ImGui::BeginMenu("Import")) {
            ImGui::MenuItem("Import Option 1");
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Export")) {
            ImGui::MenuItem("Export Option 1");
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Quit", "Ctrl+Q"))
            if (callbacks.contains("File.Quit"))
                callbacks["File.Quit"]();

        ImGui::EndMenu();
    }
}