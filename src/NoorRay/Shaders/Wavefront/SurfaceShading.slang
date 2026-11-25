import NoorRHI;
import Wavefront.Records;
import Wavefront.PathState;
import Common.Random;
import Bsdf.EnergyLut;
import Scene.SceneSurface;
import Materials.Closure;

EnergyLut sceneEnergyLut(SceneRecord scene) {
    EnergyLut lut;
    lut.tables = scene.energyTables;
    return lut;
}

//! Stochastic opacity: the same hit on the same ray always gets the same decision.
bool passesOpacity(WavefrontFrame frame, PathState path, float opacity, uint salt) {
    Random random = makeRandom(raySeed(frame, path, salt));
    return random.next() < opacity;
}
