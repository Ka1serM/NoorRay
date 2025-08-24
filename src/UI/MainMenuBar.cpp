#include "MainMenuBar.h"
#include <imgui.h>

MainMenuBar::MainMenuBar()
{
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
                invokeCallback("Add.Cube");
            }
            if (ImGui::MenuItem("Plane")) {
                invokeCallback("Add.Plane");
            }
            if (ImGui::MenuItem("Sphere")) {
                invokeCallback("Add.Sphere");
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Lights")) {
            if (ImGui::MenuItem("Sphere Light")) {
                invokeCallback("Add.SphereLight");
            }
            if (ImGui::MenuItem("Rect Light")) {
                invokeCallback("Add.RectLight");
            }
            if (ImGui::MenuItem("Disk Light")) {
                invokeCallback("Add.DiskLight");
            }
            ImGui::EndMenu();
        }

        ImGui::EndMenu();
    }
}

void MainMenuBar::renderFileMenu() {
    if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("New"))
            invokeCallback("File.New");

        if (ImGui::MenuItem("Open...", "Ctrl+O"))
            invokeCallback("File.Open");

        if (ImGui::BeginMenu("Open Recent")) {
            // Add recent files logic here
            ImGui::MenuItem("RecentFile1.MarcelScene");
            ImGui::MenuItem("RecentFile2.MarcelScene");
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Save", "Ctrl+S"))
            invokeCallback("File.Save");

        if (ImGui::MenuItem("Save As...", "Shift+Ctrl+S"))
            invokeCallback("File.SaveAs");

        if (ImGui::MenuItem("Save Copy..."))
            invokeCallback("File.SaveCopy");

        ImGui::Separator();

        if (ImGui::BeginMenu("Import"))
        {
            if (ImGui::MenuItem("Wavefront .obj"))
                invokeCallback("File.Import.Obj");

            if (ImGui::MenuItem("Chaos Camp .crtscene"))
                invokeCallback("File.Import.CrtScene");

            if (ImGui::MenuItem("Bitmap Texture"))
                invokeCallback("File.Import.Texture");

            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Export")) {
            ImGui::MenuItem("Export Option 1");
            ImGui::EndMenu();
        }

        ImGui::Separator();

        if (ImGui::MenuItem("Quit", "Ctrl+Q"))
            invokeCallback("File.Quit");

        ImGui::EndMenu();
    }
}