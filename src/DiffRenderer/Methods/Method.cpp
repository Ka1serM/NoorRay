#include "DiffRenderer/Methods/Method.h"

namespace DiffRenderer::Methods {

bool Method::supports(const Scenes::Kind kind) const {
    const Description& description = getDescription();
    switch (kind) {
    case Scenes::Kind::Scene:
        return description.scene;
    case Scenes::Kind::Object:
        return description.object;
    case Scenes::Kind::Volume:
        return description.volume;
    }
    return false;
}

}
