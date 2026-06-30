#include <iostream>
#include "NoorRay.h"

// GUI:  NoorRay
// CLI:  NoorRay <scene.json> <spp> [output.png]
int main(int argc, char* argv[]) {
    if (argc >= 3) {
        try {
            NoorRay app(argc, argv);
            app.runCli(std::stoi(argv[2]), argc >= 4 ? argv[3] : "output.exr");
        } catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    NoorRay app(1280, 720);
    app.runUi();
}
