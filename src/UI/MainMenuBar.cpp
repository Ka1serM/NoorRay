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
        renderAddMenu();
        ImGui::EndMainMenuBar();
    }
}

void MainMenuBar::renderAddMenu() {
    if (ImGui::BeginMenu("Add")) {

        if (ImGui::BeginMenu("Primitives")) {
            if (ImGui::MenuItem("Cube")) {
                if (callbacks.contains("Add.Cube"))
                    callbacks["Add.Cube"]();
            }
            if (ImGui::MenuItem("Plane")) {
                if (callbacks.contains("Add.Plane"))
                    callbacks["Add.Plane"]();
            }
            if (ImGui::MenuItem("Sphere")) {
                if (callbacks.contains("Add.Sphere"))
                    callbacks["Add.Sphere"]();
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Lights")) {
            if (ImGui::MenuItem("Sphere Light")) {
                if (callbacks.contains("Add.SphereLight"))
                    callbacks["Add.SphereLight"]();
            }
            if (ImGui::MenuItem("Rect Light")) {
                if (callbacks.contains("Add.RectLight"))
                    callbacks["Add.RectLight"]();
            }
            if (ImGui::MenuItem("Disk Light")) {
                if (callbacks.contains("Add.DiskLight"))
                    callbacks["Add.DiskLight"]();
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenu();
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
            ImGui::MenuItem("RecentFile1.MarcelScene");
            ImGui::MenuItem("RecentFile2.MarcelScene");
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
            if (ImGui::MenuItem("Wavefront .obj")) {
                if (callbacks.contains("File.Import"))
                    callbacks["File.Import"]();
            }
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