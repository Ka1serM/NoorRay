#include "Viewer.h"

extern "C" {
    void init(int width, int height);
    void close();
}

int main() {
    init(1920, 1080);
    return 0;
}