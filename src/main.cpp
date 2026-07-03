#include <iostream>
#include "NoorRay.h"
#include "Log.h"

// GUI:  NoorRay
// CLI:  NoorRay <scene.json> <spp> [output.exr]
int main(int argc, char* argv[]) {
    if (argc >= 3) {
        try {
            NoorRay app(argc, argv);
            app.runCli(std::stoi(argv[2]), argc >= 4 ? argv[3] : "output.exr");
        } catch (const std::exception& e) {
            LOG_ERROR(e.what());
            return 1;
        }
        return 0;
    }

    NoorRay app(1680, 960);
    app.runUi();
}
