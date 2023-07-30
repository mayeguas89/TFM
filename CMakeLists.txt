cmake_minimum_required(VERSION 3.26)

project(
  TFM 
  VERSION 1.0 
  DESCRIPTION "TFM Tool"
  LANGUAGES C CXX)

set(CMALE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMALE_RUNTIME_LIBRARY_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(MAXWELL REQUIRED)

add_subdirectory(externals)
add_subdirectory(mxi_viewer)
