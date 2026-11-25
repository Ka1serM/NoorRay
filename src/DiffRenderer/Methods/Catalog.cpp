#include "DiffRenderer/Methods/Catalog.h"

namespace DiffRenderer::Methods {

Method* Catalog::find(const std::string_view id) const {
    for (Method* method : all)
        if (method->getDescription().id == id)
            return method;
    return nullptr;
}

}
