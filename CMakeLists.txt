cmake_minimum_required(VERSION 3.0.2)
project(sort)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(OpenCV_INCLUDE_DIRS
  /usr/include/opencv4
)

find_package(
  OpenCV 4 REQUIRED
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(sort src/main.cpp src/track_manager.cpp src/track.cpp src/Hungarian.cpp)
target_link_libraries(sort ${catkin_LIBRARIES} ${OpenCV_LIBS})
