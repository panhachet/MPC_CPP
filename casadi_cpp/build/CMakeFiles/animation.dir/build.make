# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/robotic/NMPC/casadi_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/robotic/NMPC/casadi_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/animation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/animation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/animation.dir/flags.make

CMakeFiles/animation.dir/src/plot_animation.cpp.o: CMakeFiles/animation.dir/flags.make
CMakeFiles/animation.dir/src/plot_animation.cpp.o: ../src/plot_animation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/robotic/NMPC/casadi_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/animation.dir/src/plot_animation.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/animation.dir/src/plot_animation.cpp.o -c /home/robotic/NMPC/casadi_cpp/src/plot_animation.cpp

CMakeFiles/animation.dir/src/plot_animation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/animation.dir/src/plot_animation.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/robotic/NMPC/casadi_cpp/src/plot_animation.cpp > CMakeFiles/animation.dir/src/plot_animation.cpp.i

CMakeFiles/animation.dir/src/plot_animation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/animation.dir/src/plot_animation.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/robotic/NMPC/casadi_cpp/src/plot_animation.cpp -o CMakeFiles/animation.dir/src/plot_animation.cpp.s

# Object files for target animation
animation_OBJECTS = \
"CMakeFiles/animation.dir/src/plot_animation.cpp.o"

# External object files for target animation
animation_EXTERNAL_OBJECTS =

animation: CMakeFiles/animation.dir/src/plot_animation.cpp.o
animation: CMakeFiles/animation.dir/build.make
animation: /usr/local/lib/libcasadi.so
animation: /usr/lib/x86_64-linux-gnu/libpython3.8.so
animation: CMakeFiles/animation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/robotic/NMPC/casadi_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable animation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/animation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/animation.dir/build: animation

.PHONY : CMakeFiles/animation.dir/build

CMakeFiles/animation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/animation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/animation.dir/clean

CMakeFiles/animation.dir/depend:
	cd /home/robotic/NMPC/casadi_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/robotic/NMPC/casadi_cpp /home/robotic/NMPC/casadi_cpp /home/robotic/NMPC/casadi_cpp/build /home/robotic/NMPC/casadi_cpp/build /home/robotic/NMPC/casadi_cpp/build/CMakeFiles/animation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/animation.dir/depend

