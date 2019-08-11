cmake_minimum_required(VERSION 3.4.1)

set(jniLibs_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs)
message(STATUS "jniLibs_DIR: ${jniLibs_DIR}")

set(lib_build_DIR $ENV{HOME}/tmp)
message(STATUS "tmp_dir: ${lib_build_DIR}")

add_library(android_ceres SHARED android_ceres.cpp)

# 添加ceres库
add_library(ceres SHARED IMPORTED)
set_target_properties(ceres PROPERTIES IMPORTED_LOCATION
        ${jniLibs_DIR}/${ANDROID_ABI}/libceres.so)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ceres/internal/miniglog
)
find_library(log-lib log )

target_link_libraries(android_ceres ceres ${log-lib})