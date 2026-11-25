#include "NoorRay.App/UI/Models/OutlinerModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, OutlinerModel& state) {
    model.RegisterArray<std::vector<int>>();
    if (auto handle = model.RegisterStruct<OutlinerRow>()) {
        handle.RegisterMember("name", &OutlinerRow::name);
        handle.RegisterMember("kind", &OutlinerRow::kind);
        handle.RegisterMember("icon", &OutlinerRow::icon);
        handle.RegisterMember("depth", &OutlinerRow::depth);
        handle.RegisterMember("parent", &OutlinerRow::parent);
        handle.RegisterMember("expanded", &OutlinerRow::expanded);
        handle.RegisterMember("selected", &OutlinerRow::selected);
        handle.RegisterMember("active", &OutlinerRow::active);
        handle.RegisterMember("slots", &OutlinerRow::slots);
    }
    if (auto handle = model.RegisterStruct<MaterialCard>()) {
        handle.RegisterMember("index", &MaterialCard::index);
        handle.RegisterMember("name", &MaterialCard::name);
        handle.RegisterMember("selected", &MaterialCard::selected);
        handle.RegisterMember("nodes", &MaterialCard::nodes);
    }
    model.RegisterArray<std::vector<MaterialCard>>();
    RmlKit::UI::registerGridTypes<MaterialCard>(model);
    model.RegisterArray<std::vector<OutlinerRow>>();
    auto outliner = model.RegisterStruct<OutlinerModel>();
    outliner.RegisterMember("rows", &OutlinerModel::rows);
    outliner.RegisterMember("filter", &OutlinerModel::filter);
    outliner.RegisterMember("fileName", &OutlinerModel::fileName);
    outliner.RegisterMember("materialGrid", &OutlinerModel::materialGrid);
    outliner.RegisterMember("selectedMaterials", &OutlinerModel::selectedMaterials);
    outliner.RegisterMember("dropRow", &OutlinerModel::dropRow);
    outliner.RegisterMember("dropSlot", &OutlinerModel::dropSlot);
    outliner.RegisterMember("dropPlace", &OutlinerModel::dropPlace);
    model.Bind("outliner", &state);
}

}
