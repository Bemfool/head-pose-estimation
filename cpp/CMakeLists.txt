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
#    hdf5
#    dlib
#    opencv (for VideoCapture)
#    cminpack
#    ceres
#    m (for mathematic operations)
add_compile_options(-fPIC)

add_subdirectory(/home/keith/dlib-19.17/dlib dlib_build)

include_directories (${CMINPACK_SOURCE_DIR})
link_directories (${CMINPACK_BINARY_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(HDF5 REQUIRED C CXX)

find_package(OpenCV QUIET)
if (OpenCV_FOUND)
   message("OpenCV found.")
   include_directories(${OpenCV_INCLUDE_DIRS})
else()
   message("OpenCV not found, so we won't build the landmark-fitting-cam example.")
endif()

include_directories(./include)
add_library(hpe SHARED lib/hpe.cpp
                       lib/string_utils.cpp lib/bfm.cpp lib/random.cpp lib/transform.cpp)
target_link_libraries(hpe dlib::dlib ${CERES_LIBRARIES} cminpack m  ${OpenCV_LIBS} ${HDF5_LIBRARIES} ${HDF5_CXX_LIBRARIES})

add_executable(hpe_oneshot hpe_oneshot.cpp)
target_link_libraries(hpe_oneshot hpe)

add_executable(hpe_webcam hpe_webcam.cpp)
target_link_libraries(hpe_webcam hpe)
