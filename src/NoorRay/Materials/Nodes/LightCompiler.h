#pragma once

#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Materials/Nodes/NodeDiagnostics.h"
#include "NoorRay/Materials/Nodes/NodeTextures.h"
#include "NoorRay/Scene/Scene.h"

namespace NoorRay::Materials {

//! Sets the power, colour, falloff and graph of the light from its tree. The properties of the tree give the settings of the light: use_nodes, energy, exposure, color, use_temperature, temperature_color and normalize; the other fields of the light stay.
void compileLight(const NodeTree& tree, Scene::Light& light, NodeDiagnostics& diagnostics, NodeTextureSource& textures);

//! The environment of a world tree; last is the environment before, whose graph stays when the program does not change. The properties of the tree give the settings of the world and the film: name, use_nodes, film_transparent, sampling_method and the cycles_visibility flags.
Scene::Environment compileEnvironment(const NodeTree& tree, const Scene::Environment& last, NodeDiagnostics& diagnostics, NodeTextureSource& textures);

}
