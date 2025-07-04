#include "Viewer.h"

int main() {
    {
        Viewer viewer(1920, 1080);
        viewer.run();
    }
    Viewer::cleanup();
    return 0;
}