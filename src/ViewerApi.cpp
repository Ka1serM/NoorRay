#include "Viewer.h"

static Viewer* g_viewer = nullptr;

extern "C" {

    __declspec(dllexport) void init(int width, int height) {
        if (!g_viewer) {
            g_viewer = new Viewer(width, height);
            g_viewer->start();
        }
    }

    __declspec(dllexport) void close() {
        if (g_viewer) {
            g_viewer->stop();
            delete g_viewer;
            g_viewer = nullptr;
        }
    }
}
