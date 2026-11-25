file(GLOB_RECURSE app_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(NoorRay.App ${app_sources})
noorray_code_options(NoorRay.App)
target_compile_features(NoorRay.App PRIVATE cxx_std_23)
target_compile_definitions(NoorRay.App PRIVATE PRODUCT_NAME="NoorRay")
target_link_libraries(NoorRay.App PRIVATE NoorRay::NoorRay RmlKit::RmlKit lunasvg::lunasvg)
noorray_link_denoiser_plugins(NoorRay.App)
#! A plugin registers itself in a static initializer that nothing references, so its archive links whole.
get_property(plugins GLOBAL PROPERTY NOORRAY_PLUGINS)
get_property(plugin_resources GLOBAL PROPERTY NOORRAY_PLUGIN_RESOURCES)
foreach(plugin IN LISTS plugins)
    target_link_libraries(NoorRay.App PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${plugin}>")
endforeach()
#! The plugins load these shared libraries from the folder of the executable.
get_property(plugin_libraries GLOBAL PROPERTY NOORRAY_PLUGIN_LIBRARIES)
foreach(library IN LISTS plugin_libraries)
    add_custom_command(TARGET NoorRay.App POST_BUILD COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${library}" "$<TARGET_FILE_DIR:NoorRay.App>")
endforeach()
set_target_properties(NoorRay.App PROPERTIES OUTPUT_NAME NoorRay)
rmlkit_add_app(NoorRay.App
    APPLICATION NoorRay::App::Application
    APPLICATION_HEADER "NoorRay.App/Application.h"
    UI_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    ASSET_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    ICON "${CMAKE_CURRENT_SOURCE_DIR}/assets/icon.svg"
    DESKTOP_COMMENT "A spectral path tracer"
    DESKTOP_CATEGORIES "Graphics;3DGraphics;"
    RESOURCES ${plugin_resources})
