file(GLOB_RECURSE app_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(NoorRay.App ${app_sources})
noorray_code_options(NoorRay.App)
target_compile_features(NoorRay.App PRIVATE cxx_std_23)
target_compile_definitions(NoorRay.App PRIVATE PRODUCT_NAME="NoorRay")
target_link_libraries(NoorRay.App PRIVATE NoorRay::NoorRay RmlKit::RmlKit lunasvg::lunasvg)
set_target_properties(NoorRay.App PROPERTIES OUTPUT_NAME NoorRay)
rmlkit_add_app(NoorRay.App
    APPLICATION NoorRay::App::Application
    APPLICATION_HEADER "NoorRay.App/Application.h"
    UI_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    ICON "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.svg"
    DESKTOP_COMMENT "A spectral path tracer"
    DESKTOP_CATEGORIES "Graphics;3DGraphics;")
