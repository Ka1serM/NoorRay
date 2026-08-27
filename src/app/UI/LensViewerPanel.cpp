#include "LensViewerPanel.h"
#include <algorithm>
#include <cmath>
#include <imgui.h>
#include "Rendering/Camera/CameraInstance.h"
#include "Rendering/Camera/RealisticCamera.h"
#include "Scene/Scene.h"
#include "UI/MathInput.h"
namespace { RealisticCamera* active(Scene& scene) { auto* i = scene.getActiveCamera(); return i ? i->getCamera()->CastOrNullptr<RealisticCamera>() : nullptr; } }
LensViewerPanel::LensViewerPanel(std::string name, Scene& scene) : ImGuiComponent(std::move(name)), scene(scene) {}
void LensViewerPanel::retraceRays(const RealisticCamera& camera) {
    rays.clear(); settingsDirty = false; const auto& lens = camera.optics; if (!lens.surfaceCount) return;
    const float wavelengths[] = {FraunhoferRedNm, FraunhoferGreenNm, FraunhoferBlueNm}; const uint32_t colors[] = {IM_COL32(255,75,75,210), IM_COL32(90,255,110,210), IM_COL32(90,140,255,210)};
    for (int y = -1; y <= 1; ++y) for (int c = 0; c < 3; ++c) { TracedRay ray; ray.color = colors[c]; glm::vec3 origin(0.f, y * camera.getSensor().filmHeight() * .35f, 0.f); glm::vec3 direction = glm::normalize(glm::vec3(0.f, 0.f, lens.rearPupilZ) - origin); ray.points.push_back({origin.z, origin.y}); for (uint32_t i = 0; i < lens.surfaceCount; ++i) { glm::vec3 hit; if (!nr::optics::intersect(lens.surfaces[i], origin, direction, hit)) { ray.vignetted = true; break; } ray.points.push_back({hit.z, hit.y}); const glm::vec3 n = nr::optics::surfaceNormal(lens.surfaces[i], hit); glm::vec3 next; const float ni = sellmeierIor(lens.media[i].sellmeier, wavelengths[c]); const float nt = sellmeierIor(lens.media[lens.surfaces[i].mediumAfter].sellmeier, wavelengths[c]); ray.hits.push_back({{hit.z,hit.y},{n.z,n.y},{}}); if (!nr::optics::refract(direction,n,ni,nt,next)) { ray.vignetted=true; break; } direction=next; origin=hit; ray.hits.back().outgoing={direction.z,direction.y}; } ray.points.push_back({origin.z + direction.z * 10.f, origin.y + direction.y * 10.f}); rays.push_back(std::move(ray)); }
}
void LensViewerPanel::drawCanvas(const RealisticCamera* camera) {
    const ImVec2 pos=ImGui::GetCursorScreenPos(), size(ImGui::GetContentRegionAvail().x,std::max(200.f,ImGui::GetContentRegionAvail().y)); ImGui::InvisibleButton("##Lens",size); auto* draw=ImGui::GetWindowDrawList(); draw->AddRectFilled(pos,{pos.x+size.x,pos.y+size.y},IM_COL32(24,24,26,255)); if (!camera) return; float zmax=1.f,hmax=1.f; for(uint32_t i=0;i<camera->optics.surfaceCount;++i){zmax=std::max(zmax,camera->optics.surfaces[i].z);hmax=std::max(hmax,camera->optics.surfaces[i].apertureRadius);} const auto screen=[&](glm::vec2 p){return ImVec2(pos.x+30.f+p.x/zmax*(size.x-60.f),pos.y+size.y*.5f-p.y/hmax*(size.y*.42f));}; draw->AddLine(screen({0,0}),screen({zmax,0}),IM_COL32(90,90,90,255)); for(uint32_t i=0;i<camera->optics.surfaceCount;++i){const auto&s=camera->optics.surfaces[i];draw->AddLine(screen({s.z,-s.apertureRadius}),screen({s.z,s.apertureRadius}),s.isStop?IM_COL32(255,255,180,255):IM_COL32(120,190,255,220),2.f);} for(const auto&r:rays)for(size_t i=1;i<r.points.size();++i)draw->AddLine(screen(r.points[i-1]),screen(r.points[i]),r.color,1.f);
}
void LensViewerPanel::renderUi() { if(!ImGui::Begin(name.c_str())) { ImGui::End(); return; } auto* camera=active(scene); if(!camera){ImGui::TextDisabled("No realistic camera active.");ImGui::End();return;} if(MathInput::DragInt("Pixel stride",&pixelStride,1,1,10000))settingsDirty=true; if(settingsDirty||camera->consumeOpticsDirty())retraceRays(*camera); drawCanvas(camera); ImGui::End(); }
