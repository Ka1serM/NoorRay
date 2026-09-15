#include <catch2/catch_test_macros.hpp>

#include "NoorRaySession.h"
#include "Mesh/Assets/Mesh.h"
#include "Scene/MeshInstance.h"

// The editor keeps rendering while MaterialX compiles in the background, so a
// frame can reference a material whose program is not ready yet. Every other
// e2e test drives the CLI, which calls compileAndWait() first and therefore
// never exercises this window.
namespace
{

void addPrimitive(noorray::NoorRaySession& session, Mesh mesh, const char* instanceName)
{
    Scene& scene = session.scene();
    Mesh* asset = scene.add(std::move(mesh));
    auto instance = std::make_unique<MeshInstance>(
        scene, instanceName, asset, Transform(glm::vec3(0, 0, 0)));
    scene.setActiveObject(scene.add(std::move(instance)));
}

void renderOneFrame(noorray::NoorRaySession& session)
{
    session.pollNativeScene();
    session.updateNativeCamera();
    session.commit();
    session.render(0, 0);
    session.synchronize();
}

}

TEST_CASE("an object added in the editor renders before its material compiles",
    "[e2e][editor]")
{
    noorray::NoorRaySession session;
    session.initializeHeadlessRenderer(64, 64);

    // Deliberately no compileAndWait(): adding the object adds a material, and
    // the hit shaders index the material pointer table by the surface's
    // material index. A table not rebuilt for the new material is read out of
    // bounds and its garbage dereferenced as a device pointer, hanging the GPU.
    addPrimitive(session, Mesh::CreateSphere(session.scene(), "Sphere", {}, 24, 48),
        "Sphere Instance");
    renderOneFrame(session);

    // A second object added on top of the first must extend the table again.
    addPrimitive(session, Mesh::CreateCube(session.scene(), "Cube", {}), "Cube Instance");
    renderOneFrame(session);

    // And the same frame must still be correct once the programs are ready.
    session.rebuildNativeMaterials();
    renderOneFrame(session);
    SUCCEED("rendered across the whole compile window without a device hang");
}
