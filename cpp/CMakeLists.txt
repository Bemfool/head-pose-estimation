# Project name: head-pose-estimation
# (HPE is short for Head Pose Estimation)
project(head-pose-estimation)
cmake_minimum_required(VERSION 2.8.12)

# Executable files:
#    hpe_oneshot  
#    hpe_webcam
#
# Link libraries:
#    bfm (for manipulating Basel Face Model)
#        - hdf5
#        - dlib
#        - opencv (for VideoCapture)
#        - ceres
#        - m (for mathematic operations)
#    dlib
#    ceres

add_compile_options(-fPIC)

# Dlib
add_subdirectory(/home/keith/dlib-19.17/dlib dlib_build)

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Bfm
add_subdirectory(lib/bfm)
include_directories(./lib/bfm/include)

# Head files
include_directories(./include)

# Source files
add_library(hpe SHARED src/hpe.cpp src/string_utils.cpp)
target_link_libraries(hpe dlib::dlib ${CERES_LIBRARIES} bfm)

# Executable - hpe_oneshot
add_executable(hpe_oneshot hpe_oneshot.cpp)
target_link_libraries(hpe_oneshot hpe)

# Executable - hpe_webcam
add_executable(hpe_webcam hpe_webcam.cpp)
target_link_libraries(hpe_webcam hpe)

# Executable - time_recorder
add_executable(time_recorder time_recorder.cpp)
target_link_libraries(time_recorder hpe)
