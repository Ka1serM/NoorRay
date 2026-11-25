#include "DiffRenderer/Algorithms/Catalog.h"

namespace DiffRenderer::Algorithms {

Algorithm* Catalog::find(const std::string_view id) const {
    for (Algorithm* algorithm : all)
        if (algorithm->getDescription().id == id)
            return algorithm;
    return nullptr;
}

}
