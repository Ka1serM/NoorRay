#pragma once

class SceneObject;
class MeshInstance;
class GaussianInstance;
class LightInstance;
class CameraInstance;

namespace object_ui
{
bool render(SceneObject& object);
bool render(MeshInstance& instance);
bool render(GaussianInstance& instance);
bool render(LightInstance& instance);
bool render(CameraInstance& instance);
}

namespace domain_ui
{
bool render(SceneObject& object);
}
