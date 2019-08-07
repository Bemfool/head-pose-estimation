# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(jniLibs_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs)
message(STATUS "jniLibs_DIR: ${jniLibs_DIR}")

set(lib_build_DIR $ENV{HOME}/tmp)
message(STATUS "tmp_dir: ${lib_build_DIR}")
# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        android_ceres

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        android_ceres.cpp)

# 添加ceres库
add_library(ceres SHARED IMPORTED)
set_target_properties(ceres PROPERTIES IMPORTED_LOCATION
        ${jniLibs_DIR}/${ANDROID_ABI}/libceres.so)


include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/ceres/internal/miniglog
)

target_link_libraries( # Specifies the target library.
        android_ceres
        ceres)