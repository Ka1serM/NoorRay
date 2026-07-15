#pragma once

class SceneObject;
class MeshInstance;
class GaussianInstance;
class LightInstance;
class CameraInstance;

class SceneObjectVisitor
{
public:
    virtual ~SceneObjectVisitor() = default;
    virtual void visit(SceneObject& object) = 0;
    virtual void visit(MeshInstance& instance) = 0;
    virtual void visit(GaussianInstance& instance) = 0;
    virtual void visit(LightInstance& instance) = 0;
    virtual void visit(CameraInstance& instance) = 0;
};
