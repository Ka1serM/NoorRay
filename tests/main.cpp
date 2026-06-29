#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <string>

static int test(const std::string& name, const std::string& scene, int spp,
                const std::string& output)
{
    std::string cmd = "./NoorRay \"" + scene + "\" " + std::to_string(spp)
                    + " \"" + output + "\" 2>&1";
    std::cout << "  " << name << ": " << cmd << "\n";

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "  FAIL (" << name << "): NoorRay exited with code "
                  << ret << "\n";
        return 1;
    }

    FILE* f = std::fopen(output.c_str(), "rb");
    if (!f) {
        std::cerr << "  FAIL (" << name << "): output file not found\n";
        return 1;
    }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fclose(f);

    if (size < 16) {
        std::cerr << "  FAIL (" << name << "): output file too small ("
                  << size << " bytes)\n";
        return 1;
    }

    std::cout << "  PASS (" << name << ")\n";
    return 0;
}

int main()
{
    int failures = 0;
    std::cout << "--- NoorRay end-to-end tests ---\n";

    failures += test("simple_sphere",
                     TEST_SCENE_DIR "/scenes/simple_sphere.json",
                     4, "simple_sphere.hdr");

    std::cout << "--- " << (failures ? "FAILED" : "PASSED")
              << " (" << failures << " failures) ---\n";
    return failures;
}
