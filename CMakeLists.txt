cmake_minimum_required(VERSION 3.26)

project(
  TFM 
  VERSION 1.0 
  DESCRIPTION "TFM Tool"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)
set(CMAKE_RUNTIME_LIBRARY_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<CONFIG>)

if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (Visual Studio)
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CONFIG}")
  endforeach()
endif()

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MAXWELL REQUIRED)

add_subdirectory(externals)
add_subdirectory(mxi_viewer)
add_subdirectory(lens_raytracer)
add_subdirectory(parallel_lens_raytracer)

