#include "Viewer.h"

static Viewer* g_viewer = nullptr;

extern "C" {

    // Singleton init that returns the HWND (handle to the native window)
    __declspec(dllexport) void* init(int width, int height) {
        if (!g_viewer) {
            g_viewer = new Viewer(width, height);
            g_viewer->start();
        }
        return g_viewer->getHwnd();
    }

    __declspec(dllexport) void close() {
        if (g_viewer) {
            g_viewer->stop();
            delete g_viewer;
            g_viewer = nullptr;
        }
    }

    __declspec(dllexport) void add_mesh(Vertex* vertices, int vertexCount, uint32_t* indices, int indexCount) {
        if (!g_viewer || !vertices || !indices || vertexCount <= 0 || indexCount <= 0)
            return;

        std::vector<Face> faces(indexCount / 3, Face{0, 0, 0, 0});
        MeshData mesh;
        mesh.vertices.assign(vertices, vertices + vertexCount);
        mesh.indices.assign(indices, indices + indexCount);
        mesh.faces = std::move(faces);

        g_viewer->addMesh(std::move(mesh));
    }

}
