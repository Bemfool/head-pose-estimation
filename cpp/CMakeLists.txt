# Project name: landmark-fitting
project(landmark-fitting)
cmake_minimum_required(VERSION 2.8.12)

# Three executable file:
#    landmark-fitting-cminpack: landmark fitting using cminpack (BUG)
#    landmark-fitting-ceres: landmark fitting using Ceres with one picture
#    landmark-fitting-cam: landmark fitting using Ceres with camera
# Link libraries:
#    dlib
#    opencv (for VideoCapture)
#    cminpack
#    ceres
#    m (for mathematic operations)

add_subdirectory(/home/keith/dlib-19.17/dlib dlib_build)

add_executable(landmark-fitting-cminpack landmark-fitting-cminpack.cpp)
target_link_libraries(landmark-fitting-cminpack dlib::dlib)

include_directories (${CMINPACK_SOURCE_DIR})
link_directories (${CMINPACK_BINARY_DIR})
target_link_libraries (landmark-fitting-cminpack cminpack)
target_link_libraries (landmark-fitting-cminpack m)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(landmark-fitting-ceres landmark-fitting-ceres.cpp)
target_link_libraries(landmark-fitting-ceres dlib::dlib)
target_link_libraries(landmark-fitting-ceres ${CERES_LIBRARIES})

add_executable(landmark-fitting-cam landmark-fitting-cam.cpp)

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
   target_link_libraries(landmark-fitting-cam dlib::dlib ${CERES_LIBRARIES} ${OpenCV_LIBS} )
else()
   message("OpenCV not found, so we won't build the landmark-fitting-cam example.")
endif()


