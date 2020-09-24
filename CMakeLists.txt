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

OPTION(SHUT_UP "Close output" OFF)
IF(SHUT_UP)
    add_definitions("-DHPE_SHUT_UP")
    add_definitions("-DBFM_SHUT_UP")
ENDIF(SHUT_UP)

add_definitions(
    "-DUSE_AVX_INSTRUCTIONS"
    "-DUSE_SSE2_INSTRUCTIONS"
    "-DUSE_SSE4_INSTRUCTIONS"
    "-DCMAKE_BUILD_TYPE=Release"
)

add_library(hpe SHARED src/hpe_problem.cpp src/string_utils.cpp)
target_compile_features(hpe PRIVATE cxx_std_17)

# OpenCV
set(OpenCV_DIR "/usr/local/opencv3/share/OpenCV")   
find_package(OpenCV REQUIRED)
target_include_directories(hpe PUBLIC ${OpenCV_INCLUDE_DIRS})

# Dlib
add_subdirectory(/home/bemfoo/Repository/dlib dlib_build)

# Ceres
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# Hdf5
find_package(HDF5 REQUIRED C CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Bfm
add_subdirectory(lib/BFM-tools)
include_directories(./lib/BFM-tools/include)

# GLM
include_directories(/home/bemfoo/Repository/glm)

# OpenGL
include_directories(/home/bemfoo/Repository/glad/include)
include_directories(/home/bemfoo/Repository/glfw/include)

# stb_image.h
include_directories(/home/bemfoo/Repository/stb)

# tinyxml2
include_directories(/home/bemfoo/Repository/tinyxml2)

# Head files
include_directories(./include)

# Source files
target_link_libraries(hpe dlib::dlib ${CERES_LIBRARIES} bfm_manager)

OPTION(GEN_HPE_SAMPLES "Generate samples." ON)
if(GEN_HPE_SAMPLES)
    # Executable - hpe_oneshot  
    add_executable(hpe_oneshot hpe_oneshot.cpp)
    target_link_libraries(hpe_oneshot hpe)

    # Executable - hpe_webcam
    add_executable(hpe_webcam hpe_webcam.cpp)
    target_link_libraries(hpe_webcam hpe)

    add_executable(hpe_oneshot_openGL hpe_oneshot_openGL.cpp lib/glad/glad.c)
    target_link_libraries(hpe_oneshot_openGL GL glfw3
                                             m Xrandr Xi X11 Xxf86vm pthread dl Xinerama Xcursor
                                             hpe)

    add_executable(hpe_webcam_openGL hpe_webcam_openGL.cpp lib/glad/glad.c)
    target_link_libraries(hpe_webcam_openGL GL glfw3
                                            m Xrandr Xi X11 Xxf86vm pthread dl Xinerama Xcursor
                                            hpe)
   
endif()