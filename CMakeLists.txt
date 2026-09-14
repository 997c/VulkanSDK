cmake_minimum_required(VERSION 3.20)
project(Torus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC does not have a /std:c++26 switch yet - C++26 is enabled via
# /std:c++latest (reports __cplusplus = 202400 with /Zc:__cplusplus)
if(MSVC)
    unset(CMAKE_CXX_STANDARD)
    unset(CMAKE_CXX_STANDARD_REQUIRED)
    add_compile_options(/std:c++latest /Zc:__cplusplus)
endif()

# Vulkan SDK 1.4.357.0 (no longer bundles GLFW - SDL3 is the windowing library
# shipped with the SDK, same one used by the SDK's own tutorials/samples)
set(VULKAN_SDK_DIR "C:/VulkanSDK/1.4.357.0")
set(VULKAN_INCLUDE_DIRS ${VULKAN_SDK_DIR}/Include)
set(VULKAN_LIBRARY ${VULKAN_SDK_DIR}/Lib/vulkan-1.lib)
set(SDL3_INCLUDE_DIR ${VULKAN_SDK_DIR}/Include/SDL3)
set(SDL3_LIBRARY ${VULKAN_SDK_DIR}/Lib/SDL3.lib)
set(SDL3_DLL ${VULKAN_SDK_DIR}/Bin/SDL3.dll)

# Shader compilation
set(SHADER_DIR ${CMAKE_SOURCE_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_DIR})

set(GLSLANG_VALIDATOR ${VULKAN_SDK_DIR}/Bin/glslangValidator.exe)

# Compile combined vertex shader (mode selected at runtime via push constants)
add_custom_command(
    OUTPUT ${SHADER_DIR}/vert.vert.spv
    COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_DIR}/vert.vert -o ${SHADER_DIR}/vert.vert.spv
    DEPENDS ${SHADER_DIR}/vert.vert
    COMMENT "Compiling vertex shader from vert.vert"
)

# Compile combined fragment shader
add_custom_command(
    OUTPUT ${SHADER_DIR}/frag.frag.spv
    COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER_DIR}/frag.frag -o ${SHADER_DIR}/frag.frag.spv
    DEPENDS ${SHADER_DIR}/frag.frag
    COMMENT "Compiling fragment shader from frag.frag"
)

add_custom_target(Shaders ALL DEPENDS
    ${SHADER_DIR}/vert.vert.spv
    ${SHADER_DIR}/frag.frag.spv
)

add_executable(Torus WIN32
    main.cpp
    main.h
)
target_include_directories(Torus PRIVATE ${VULKAN_INCLUDE_DIRS})
target_link_libraries(Torus ${VULKAN_LIBRARY} ${SDL3_LIBRARY})
add_dependencies(Torus Shaders)

# SDL3 is a DLL - make sure it sits next to Torus.exe
add_custom_command(TARGET Torus POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SDL3_DLL} $<TARGET_FILE_DIR:Torus>/SDL3.dll
)

# Copy shaders to build directory so the app can find them
add_custom_command(TARGET Torus POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SHADER_DIR}/vert.vert.spv $<TARGET_FILE_DIR:Torus>/shaders/vert.vert.spv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${SHADER_DIR}/frag.frag.spv $<TARGET_FILE_DIR:Torus>/shaders/frag.frag.spv
)