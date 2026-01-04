cmake_minimum_required(VERSION 3.0.2)
project(carvision)

## -----------------------------------------------------------------------------
## Catkin deps
## -----------------------------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  std_msgs
  morai_msgs
  tf
)

## -----------------------------------------------------------------------------
## GeographicLib 수동 탐색 (libgeographic-dev)
## -----------------------------------------------------------------------------
find_path(GEOGRAPHIC_INCLUDE_DIR GeographicLib/LocalCartesian.hpp)
find_library(GEOGRAPHIC_LIBRARY NAMES Geographic GeographicLib)

if(NOT GEOGRAPHIC_INCLUDE_DIR OR NOT GEOGRAPHIC_LIBRARY)
  message(FATAL_ERROR
    "GeographicLib not found. Install it first:\n"
    "  sudo apt-get update && sudo apt-get install -y libgeographic-dev")
endif()

## -----------------------------------------------------------------------------
## OpenCV (YOLO DNN)
## -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)

## -----------------------------------------------------------------------------
## Catkin package
## -----------------------------------------------------------------------------
catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES carvision
  CATKIN_DEPENDS roscpp nav_msgs geometry_msgs sensor_msgs std_msgs morai_msgs tf
  # DEPENDS GeographicLib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GEOGRAPHIC_INCLUDE_DIR}   # /usr/include 등
  ${OpenCV_INCLUDE_DIRS}
)

## -----------------------------------------------------------------------------
## Executables
## -----------------------------------------------------------------------------

# Pure Pursuit + 회피 플래그 노드
add_executable(avoid_lidar_car src/decision/avoid_lidar_car.cpp)
target_compile_features(avoid_lidar_car PRIVATE cxx_std_11)
target_link_libraries(avoid_lidar_car
  ${catkin_LIBRARIES}
  ${GEOGRAPHIC_LIBRARY}
)

# LiDAR 좌표변환 노드
add_executable(convert_lidar_car src/perception/convert_lidar_car.cpp)
target_compile_features(convert_lidar_car PRIVATE cxx_std_11)
target_link_libraries(convert_lidar_car
  ${catkin_LIBRARIES}
  ${GEOGRAPHIC_LIBRARY}
)

# 장애물 검출 + /avoid_flag 퍼블리셔
add_executable(obstacle_detector_car src/perception/obstacle_detector_car.cpp)
target_compile_features(obstacle_detector_car PRIVATE cxx_std_11)
target_link_libraries(obstacle_detector_car
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#
add_executable(avoid_camera_car src/decision/avoid_camera_car.cpp)
target_compile_features(avoid_camera_car PRIVATE cxx_std_11)
target_link_libraries(avoid_camera_car
  ${catkin_LIBRARIES}
  ${GEOGRAPHIC_LIBRARY}
)


# (옵션) catkin 메시지 의존성 추가하고 싶으면 아래 주석 풀어도 됨
# add_dependencies(pp_fixed_speed_car      ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_dependencies(convert_lidar_car       ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# add_dependencies(obstacle_detector_car   ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
