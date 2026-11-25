#include "NoorRay.App/UI/Models/InspectorModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, InspectorModel& inspector, EnvironmentModel& environment) {
    if (auto handle = model.RegisterStruct<MaterialLink>()) {
        handle.RegisterMember("label", &MaterialLink::label);
        handle.RegisterMember("name", &MaterialLink::name);
        handle.RegisterMember("sceneIndex", &MaterialLink::sceneIndex);
    }
    model.RegisterArray<std::vector<MaterialLink>>();
    auto inspectorFields = model.RegisterStruct<InspectorModel>();
    inspectorFields.RegisterMember("tab", &InspectorModel::tab);
    inspectorFields.RegisterMember("kind", &InspectorModel::kind);
    inspectorFields.RegisterMember("heading", &InspectorModel::heading);
    inspectorFields.RegisterMember("name", &InspectorModel::name);
    inspectorFields.RegisterMember("positionX", &InspectorModel::positionX);
    inspectorFields.RegisterMember("positionY", &InspectorModel::positionY);
    inspectorFields.RegisterMember("positionZ", &InspectorModel::positionZ);
    inspectorFields.RegisterMember("rotationX", &InspectorModel::rotationX);
    inspectorFields.RegisterMember("rotationY", &InspectorModel::rotationY);
    inspectorFields.RegisterMember("rotationZ", &InspectorModel::rotationZ);
    inspectorFields.RegisterMember("scaleX", &InspectorModel::scaleX);
    inspectorFields.RegisterMember("scaleY", &InspectorModel::scaleY);
    inspectorFields.RegisterMember("scaleZ", &InspectorModel::scaleZ);
    inspectorFields.RegisterMember("meshName", &InspectorModel::meshName);
    inspectorFields.RegisterMember("triangles", &InspectorModel::triangles);
    inspectorFields.RegisterMember("vertices", &InspectorModel::vertices);
    inspectorFields.RegisterMember("visible", &InspectorModel::visible);
    inspectorFields.RegisterMember("doubleSided", &InspectorModel::doubleSided);
    inspectorFields.RegisterMember("castsShadows", &InspectorModel::castsShadows);
    inspectorFields.RegisterMember("cameraVisible", &InspectorModel::cameraVisible);
    inspectorFields.RegisterMember("materials", &InspectorModel::materials);
    inspectorFields.RegisterMember("materialOverride", &InspectorModel::materialOverride);
    inspectorFields.RegisterMember("lightType", &InspectorModel::lightType);
    inspectorFields.RegisterMember("color", &InspectorModel::color);
    inspectorFields.RegisterMember("colorOpen", &InspectorModel::colorOpen);
    inspectorFields.RegisterMember("power", &InspectorModel::power);
    inspectorFields.RegisterMember("radius", &InspectorModel::radius);
    inspectorFields.RegisterMember("width", &InspectorModel::width);
    inspectorFields.RegisterMember("height", &InspectorModel::height);
    inspectorFields.RegisterMember("spotAngle", &InspectorModel::spotAngle);
    inspectorFields.RegisterMember("spotBlend", &InspectorModel::spotBlend);
    inspectorFields.RegisterMember("angle", &InspectorModel::angle);
    inspectorFields.RegisterMember("cameraType", &InspectorModel::cameraType);
    inspectorFields.RegisterMember("activeCamera", &InspectorModel::activeCamera);
    inspectorFields.RegisterMember("fieldOfView", &InspectorModel::fieldOfView);
    inspectorFields.RegisterMember("orthographicHeight", &InspectorModel::orthographicHeight);
    inspectorFields.RegisterMember("focusDistance", &InspectorModel::focusDistance);
    inspectorFields.RegisterMember("fStop", &InspectorModel::fStop);
    inspectorFields.RegisterMember("blades", &InspectorModel::blades);
    inspectorFields.RegisterMember("sensorWidth", &InspectorModel::sensorWidth);
    inspectorFields.RegisterMember("sensorHeight", &InspectorModel::sensorHeight);
    inspectorFields.RegisterMember("lensElements", &InspectorModel::lensElements);
    inspectorFields.RegisterMember("gaussianCount", &InspectorModel::gaussianCount);
    inspectorFields.RegisterMember("shDegree", &InspectorModel::shDegree);
    auto environmentFields = model.RegisterStruct<EnvironmentModel>();
    environmentFields.RegisterMember("textureName", &EnvironmentModel::textureName);
    environmentFields.RegisterMember("color", &EnvironmentModel::color);
    environmentFields.RegisterMember("colorOpen", &EnvironmentModel::colorOpen);
    environmentFields.RegisterMember("strength", &EnvironmentModel::strength);
    environmentFields.RegisterMember("rotation", &EnvironmentModel::rotation);
    environmentFields.RegisterMember("visibleToCamera", &EnvironmentModel::visibleToCamera);
    model.Bind("inspector", &inspector);
    model.Bind("environment", &environment);
}

}
