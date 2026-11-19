#pragma once

#include <vector>

#include <RmlUi/Core/DataModelHandle.h>
#include <RmlUi/Core/Types.h>

#include "NoorRay.App/UI/Models/NumberField.h"

namespace NoorRay::App {

struct MaterialLink {
    Rml::String name;
    int sceneIndex = -1;
};

struct InspectorModel {
    Rml::String tab = "object";
    Rml::String kind;
    Rml::String heading;
    Rml::String name;

    NumberField positionX = 0.0f;
    NumberField positionY = 0.0f;
    NumberField positionZ = 0.0f;
    NumberField rotationX = 0.0f;
    NumberField rotationY = 0.0f;
    NumberField rotationZ = 0.0f;
    NumberField scaleX = 1.0f;
    NumberField scaleY = 1.0f;
    NumberField scaleZ = 1.0f;

    Rml::String meshName;
    int triangles = 0;
    int vertices = 0;
    bool visible = true;
    bool doubleSided = false;
    bool castsShadows = true;
    bool cameraVisible = true;
    std::vector<MaterialLink> materials;
    bool materialOverride = false;

    Rml::String lightType;
    NumberField colorR = 1.0f;
    NumberField colorG = 1.0f;
    NumberField colorB = 1.0f;
    NumberField power = 100.0f;
    NumberField radius = 0.0f;
    NumberField width = 1.0f;
    NumberField height = 1.0f;
    NumberField spotAngle = 45.0f;
    NumberField spotBlend = 0.15f;
    NumberField angle = 0.5f;

    Rml::String cameraType;
    bool activeCamera = false;
    NumberField fieldOfView = 40.0f;
    NumberField orthographicHeight = 10.0f;
    NumberField focusDistance = 10.0f;
    NumberField fStop = 0.0f;
    int blades = 0;
    NumberField sensorWidth = 36.0f;
    NumberField sensorHeight = 24.0f;
    int lensElements = 0;

    Rml::String gaussianCount;
    int shDegree = 0;
};

struct EnvironmentModel {
    Rml::String textureName;
    NumberField colorR = 0.0f;
    NumberField colorG = 0.0f;
    NumberField colorB = 0.0f;
    NumberField strength = 1.0f;
    NumberField rotation = 0.0f;
    bool visibleToCamera = false;
};

void bind(Rml::DataModelConstructor& model, InspectorModel& inspector, EnvironmentModel& environment);

}
