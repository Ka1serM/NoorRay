#include "Viewer.h"

extern "C" {
    void init(int width, int height);
    void close();
    void add_mesh(Vertex* vertices, int vertexCount, uint32_t* indices, int indexCount);
}

int main() {
    init(1920, 1080);
    return 0;
}