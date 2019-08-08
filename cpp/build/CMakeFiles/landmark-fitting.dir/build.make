# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/keith/Desktop/landmark-fitting

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/keith/Desktop/landmark-fitting/build

# Include any dependencies generated for this target.
include CMakeFiles/landmark-fitting.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/landmark-fitting.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/landmark-fitting.dir/flags.make

CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o: CMakeFiles/landmark-fitting.dir/flags.make
CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o: ../landmark-fitting.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/keith/Desktop/landmark-fitting/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o -c /home/keith/Desktop/landmark-fitting/landmark-fitting.cpp

CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/keith/Desktop/landmark-fitting/landmark-fitting.cpp > CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.i

CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/keith/Desktop/landmark-fitting/landmark-fitting.cpp -o CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.s

CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.requires:

.PHONY : CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.requires

CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.provides: CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.requires
	$(MAKE) -f CMakeFiles/landmark-fitting.dir/build.make CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.provides.build
.PHONY : CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.provides

CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.provides.build: CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o


# Object files for target landmark-fitting
landmark__fitting_OBJECTS = \
"CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o"

# External object files for target landmark-fitting
landmark__fitting_EXTERNAL_OBJECTS =

landmark-fitting: CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o
landmark-fitting: CMakeFiles/landmark-fitting.dir/build.make
landmark-fitting: dlib_build/libdlib.a
landmark-fitting: /usr/lib/x86_64-linux-gnu/libnsl.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libSM.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libICE.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libX11.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libXext.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libpng.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libz.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libjpeg.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libopenblas.so
landmark-fitting: /usr/lib/x86_64-linux-gnu/libsqlite3.so
landmark-fitting: CMakeFiles/landmark-fitting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/keith/Desktop/landmark-fitting/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable landmark-fitting"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/landmark-fitting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/landmark-fitting.dir/build: landmark-fitting

.PHONY : CMakeFiles/landmark-fitting.dir/build

CMakeFiles/landmark-fitting.dir/requires: CMakeFiles/landmark-fitting.dir/landmark-fitting.cpp.o.requires

.PHONY : CMakeFiles/landmark-fitting.dir/requires

CMakeFiles/landmark-fitting.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/landmark-fitting.dir/cmake_clean.cmake
.PHONY : CMakeFiles/landmark-fitting.dir/clean

CMakeFiles/landmark-fitting.dir/depend:
	cd /home/keith/Desktop/landmark-fitting/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/keith/Desktop/landmark-fitting /home/keith/Desktop/landmark-fitting /home/keith/Desktop/landmark-fitting/build /home/keith/Desktop/landmark-fitting/build /home/keith/Desktop/landmark-fitting/build/CMakeFiles/landmark-fitting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/landmark-fitting.dir/depend
