#include "RenderTestFixture.h"
#include "FurnaceTestSupport.h"

#include <catch2/catch_test_macros.hpp>

class OpaqueFurnaceTest : public RenderTestFixture {};

TEST_CASE_METHOD(OpaqueFurnaceTest, "white furnace conserves energy", "[e2e][furnace]")
{
    const std::string output = render("white_furnace.json", 4096, "white_furnace.exr");
    checkEnergyConservation(BitmapReader::read(output));
}
