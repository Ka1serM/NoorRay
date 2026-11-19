#include "NoorRay.App/UI/Models/OutlinerModel.h"

namespace NoorRay::App {

void bind(Rml::DataModelConstructor& model, OutlinerModel& state) {
    if (auto handle = model.RegisterStruct<OutlinerRow>()) {
        handle.RegisterMember("name", &OutlinerRow::name);
        handle.RegisterMember("kind", &OutlinerRow::kind);
        handle.RegisterMember("icon", &OutlinerRow::icon);
        handle.RegisterMember("depth", &OutlinerRow::depth);
        handle.RegisterMember("parent", &OutlinerRow::parent);
        handle.RegisterMember("expanded", &OutlinerRow::expanded);
        handle.RegisterMember("selected", &OutlinerRow::selected);
        handle.RegisterMember("active", &OutlinerRow::active);
        handle.RegisterMember("material", &OutlinerRow::material);
    }
    model.RegisterArray<std::vector<OutlinerRow>>();
    auto outliner = model.RegisterStruct<OutlinerModel>();
    outliner.RegisterMember("rows", &OutlinerModel::rows);
    outliner.RegisterMember("filter", &OutlinerModel::filter);
    outliner.RegisterMember("fileName", &OutlinerModel::fileName);
    outliner.RegisterMember("materials", &OutlinerModel::materials);
    outliner.RegisterMember("dropRow", &OutlinerModel::dropRow);
    outliner.RegisterMember("dropPlace", &OutlinerModel::dropPlace);
    model.Bind("outliner", &state);
}

}
