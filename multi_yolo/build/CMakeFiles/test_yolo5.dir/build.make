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
CMAKE_SOURCE_DIR = /home/qinhua/work/multi_yolo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qinhua/work/multi_yolo/build

# Include any dependencies generated for this target.
include CMakeFiles/test_yolo5.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test_yolo5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_yolo5.dir/flags.make

CMakeFiles/test_yolo5.dir/main.cpp.o: CMakeFiles/test_yolo5.dir/flags.make
CMakeFiles/test_yolo5.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qinhua/work/multi_yolo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test_yolo5.dir/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_yolo5.dir/main.cpp.o -c /home/qinhua/work/multi_yolo/main.cpp

CMakeFiles/test_yolo5.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_yolo5.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qinhua/work/multi_yolo/main.cpp > CMakeFiles/test_yolo5.dir/main.cpp.i

CMakeFiles/test_yolo5.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_yolo5.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qinhua/work/multi_yolo/main.cpp -o CMakeFiles/test_yolo5.dir/main.cpp.s

CMakeFiles/test_yolo5.dir/test.cpp.o: CMakeFiles/test_yolo5.dir/flags.make
CMakeFiles/test_yolo5.dir/test.cpp.o: ../test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qinhua/work/multi_yolo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/test_yolo5.dir/test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_yolo5.dir/test.cpp.o -c /home/qinhua/work/multi_yolo/test.cpp

CMakeFiles/test_yolo5.dir/test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_yolo5.dir/test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qinhua/work/multi_yolo/test.cpp > CMakeFiles/test_yolo5.dir/test.cpp.i

CMakeFiles/test_yolo5.dir/test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_yolo5.dir/test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qinhua/work/multi_yolo/test.cpp -o CMakeFiles/test_yolo5.dir/test.cpp.s

CMakeFiles/test_yolo5.dir/yolo.cpp.o: CMakeFiles/test_yolo5.dir/flags.make
CMakeFiles/test_yolo5.dir/yolo.cpp.o: ../yolo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qinhua/work/multi_yolo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/test_yolo5.dir/yolo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_yolo5.dir/yolo.cpp.o -c /home/qinhua/work/multi_yolo/yolo.cpp

CMakeFiles/test_yolo5.dir/yolo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_yolo5.dir/yolo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qinhua/work/multi_yolo/yolo.cpp > CMakeFiles/test_yolo5.dir/yolo.cpp.i

CMakeFiles/test_yolo5.dir/yolo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_yolo5.dir/yolo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qinhua/work/multi_yolo/yolo.cpp -o CMakeFiles/test_yolo5.dir/yolo.cpp.s

# Object files for target test_yolo5
test_yolo5_OBJECTS = \
"CMakeFiles/test_yolo5.dir/main.cpp.o" \
"CMakeFiles/test_yolo5.dir/test.cpp.o" \
"CMakeFiles/test_yolo5.dir/yolo.cpp.o"

# External object files for target test_yolo5
test_yolo5_EXTERNAL_OBJECTS =

test_yolo5: CMakeFiles/test_yolo5.dir/main.cpp.o
test_yolo5: CMakeFiles/test_yolo5.dir/test.cpp.o
test_yolo5: CMakeFiles/test_yolo5.dir/yolo.cpp.o
test_yolo5: CMakeFiles/test_yolo5.dir/build.make
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_dnn.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_freetype.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_gapi.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_ml.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_objdetect.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_photo.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_stitching.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_video.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/libsophon-current/lib/libbmlib.so
test_yolo5: /opt/sophon/libsophon-current/lib/libbmrt.so
test_yolo5: /opt/sophon/sophon-ffmpeg-latest/lib/libavcodec.so
test_yolo5: /opt/sophon/sophon-ffmpeg-latest/lib/libavformat.so
test_yolo5: /opt/sophon/sophon-ffmpeg-latest/lib/libavutil.so
test_yolo5: /opt/sophon/sophon-ffmpeg-latest/lib/libavfilter.so
test_yolo5: /opt/sophon/sophon-ffmpeg-latest/lib/libavdevice.so
test_yolo5: /opt/sophon/sophon-ffmpeg-latest/lib/libswscale.so
test_yolo5: /opt/sophon/sophon-ffmpeg-latest/lib/libswresample.so
test_yolo5: /opt/sophon/libsophon-current/lib/libbmcv.so
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_calib3d.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_features2d.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_flann.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_highgui.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_videoio.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_imgcodecs.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_imgproc.so.4.1.0-sophon-0.5.1
test_yolo5: /opt/sophon/sophon-opencv-latest/lib/libopencv_core.so.4.1.0-sophon-0.5.1
test_yolo5: CMakeFiles/test_yolo5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qinhua/work/multi_yolo/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable test_yolo5"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_yolo5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_yolo5.dir/build: test_yolo5

.PHONY : CMakeFiles/test_yolo5.dir/build

CMakeFiles/test_yolo5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_yolo5.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_yolo5.dir/clean

CMakeFiles/test_yolo5.dir/depend:
	cd /home/qinhua/work/multi_yolo/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qinhua/work/multi_yolo /home/qinhua/work/multi_yolo /home/qinhua/work/multi_yolo/build /home/qinhua/work/multi_yolo/build /home/qinhua/work/multi_yolo/build/CMakeFiles/test_yolo5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_yolo5.dir/depend

