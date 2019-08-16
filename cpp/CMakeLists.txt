# Project name: head-pose-estimation
# (HPE is short for Head Pose Estimation)
project(head-pose-estimation)
cmake_minimum_required(VERSION 2.8.12)

# Five executable file:
#    Name Format: hpe-[oneshot|cam]-[cminpack|ceres]-{numericdiff|analyticdiff}
#
#    file                          | picture source   | solver tool | diff method  | available
#    ------------------------------|------------------|-------------|--------------|------------
#    hpe-oneshot-cminpack          | one picture url  | minpack     | numeric diff | no
#    hpe-oneshot-ceres-numericdiff | one picture url  | ceres       | numeric diff | yes
#    hpe-cam-ceres-numericdiff     | camera           | ceres       | numeric diff | yes
#    hpe-cam-ceres-analyticdiff        | camera           | ceres       | analytic diff    | no
#    hpe-oneshot-ceres-analyticdiff    | one picture url  | ceres       | analytic diff    | no
#
# Link libraries:
#    dlib
#    opencv (for VideoCapture)
#    cminpack
#    ceres
#    m (for mathematic operations)

add_subdirectory(/home/keith/dlib-19.17/dlib dlib_build)

add_executable(hpe-oneshot-cminpack hpe-oneshot-cminpack.cpp)
target_link_libraries(hpe-oneshot-cminpack dlib::dlib)

include_directories (${CMINPACK_SOURCE_DIR})
link_directories (${CMINPACK_BINARY_DIR})
target_link_libraries (hpe-oneshot-cminpack cminpack)
target_link_libraries (hpe-oneshot-cminpack m)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(hpe-oneshot-ceres-numericdiff hpe-oneshot-ceres-numericdiff.cpp)
target_link_libraries(hpe-oneshot-ceres-numericdiff dlib::dlib)
target_link_libraries(hpe-oneshot-ceres-numericdiff ${CERES_LIBRARIES})

add_executable(hpe-oneshot-ceres-analyticdiff hpe-oneshot-ceres-analyticdiff.cpp)
target_link_libraries(hpe-oneshot-ceres-analyticdiff dlib::dlib)
target_link_libraries(hpe-oneshot-ceres-analyticdiff ${CERES_LIBRARIES})


add_executable(hpe-cam-ceres-numericdiff hpe-cam-ceres-numericdiff.cpp)
add_executable(hpe-cam-ceres-analyticdiff hpe-cam-ceres-analyticdiff.cpp)

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
   include_directories(${OpenCV_INCLUDE_DIRS})
   target_link_libraries(hpe-cam-ceres-numericdiff dlib::dlib ${CERES_LIBRARIES} ${OpenCV_LIBS} )
   target_link_libraries(hpe-cam-ceres-analyticdiff dlib::dlib ${CERES_LIBRARIES} ${OpenCV_LIBS} )
else()
   message("OpenCV not found, so we won't build the landmark-fitting-cam example.")
endif()


