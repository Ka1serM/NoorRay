#pragma once

#include <string>
#include <vector>

#include "NoorRay/Core/Math.h"
#include "NoorRay/Materials/NodeTree.h"
#include "NoorRay/Scene/Texture.h"

namespace NoorRay::Materials {

//! The foldSide x foldSide texels of a texture in linear colour, from the bottom row; empty for a block format or a texture without data.
std::vector<Core::float4> sampleImage(const Scene::Texture& texture);
//! Gives an image or environment texture node without an image the image, and the texture records of its uses.
void setNodeImage(NodeTree& tree, std::uint32_t node, const std::string& image, Core::Handle<Scene::Texture> texture);

}
