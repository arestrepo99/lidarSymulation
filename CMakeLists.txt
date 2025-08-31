#cmake_minimum_required(VERSION 3.16)
#project(LidarDenoiser)
#
#set(CMAKE_C_STANDARD 11)
#set(CMAKE_CXX_STANDARD 14)
#
## Find LibTorch
#find_package(Torch REQUIRED)
## Find SDL2
#find_package(SDL2 REQUIRED)
#
## Include directories
#include_directories(
#    src/headers
#    ${SDL2_INCLUDE_DIRS}
#    ${TORCH_INCLUDE_DIRS}
#)
#
## Set output directories
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
#
## C Executables
#add_executable(generateData
#    src/c/data_generator.c
#    src/c/lidar_system.c
#)
#target_link_libraries(generateData ${SDL2_LIBRARIES} m)
#
#add_executable(simulation
#    src/c/main.c
#    src/c/lidar_system.c
#    src/c/rendering.c
#)
#target_link_libraries(simulation ${SDL2_LIBRARIES} m)
#
#add_executable(test_c
#    src/c/test_c.c
#)
#target_link_libraries(test_c m)
#
## C++ Executables
#add_executable(lidar_denoiser_main
#    src/cpp/main.cpp
#    src/cpp/lidar_denoiser.cpp
#    src/cpp/lidar_denoiser_c.cpp
#)
#target_link_libraries(lidar_denoiser_main ${TORCH_LIBRARIES} ${SDL2_LIBRARIES} m)
#
## Create shared library for C interface
#add_library(lidar_denoiser SHARED
#    src/cpp/lidar_denoiser.cpp
#    src/cpp/lidar_denoiser_c.cpp
#)
#target_link_libraries(lidar_denoiser ${TORCH_LIBRARIES})
#
## Shape renderer executable (if you have this)
#add_executable(shape_renderer
#    src/c/shape_renderer.c
#)
#target_link_libraries(shape_renderer ${SDL2_LIBRARIES} m)
#
## For macOS, set the deployment target
#if(APPLE)
#    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS version")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
#endif()
#
## Custom targets for convenience
#add_custom_target(generateDataTarget DEPENDS generateData)
#add_custom_target(simulationTarget DEPENDS simulation)
#add_custom_target(allTarget DEPENDS generateData simulation lidar_denoiser_main lidar_denoiser)
#
## Print help message
#message("
#Build targets available:
#  make generateDataTarget    - Build data generator
#  make simulationTarget      - Build simulation
#  make lidar_denoiser_main   - Build C++ main executable
#  make lidar_denoiser        - Build shared library
#  make allTarget             - Build all targets
#")