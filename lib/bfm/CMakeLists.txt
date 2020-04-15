# Project name: bfm-tools
# (BFM is short for Basel Face Model)
project(bfm-tools)
cmake_minimum_required(VERSION 2.8.12)

add_compile_options(-fPIC)

add_subdirectory(/home/keith/dlib-19.17/dlib dlib_build)
find_package(HDF5 REQUIRED C CXX)

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
   message("OpenCV found.")
   include_directories(${OpenCV_INCLUDE_DIRS})
else()
   message("OpenCV not found, so we won't build the landmark-fitting-cam example.")
endif()

include_directories(./include)
add_library(bfm SHARED src/bfm.cpp src/random.cpp)
target_link_libraries(bfm dlib::dlib m  ${OpenCV_LIBS} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

add_executable(bfm-ex bfm-ex.cpp)
target_link_libraries(bfm-ex bfm)
