cmake_minimum_required(VERSION 3.0.2)
project(roscpp_morai)

## -----------------------------------------------------------------------------
## Catkin deps
## -----------------------------------------------------------------------------
find_package(catkin REQUIRED COMPONENTS
  roscpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  morai_msgs
  rosbag
  tf
  tf2_ros
  tf2_sensor_msgs
)

## -----------------------------------------------------------------------------
## Geographic(lib) – config 파일이 없는 환경 대비: 수동 탐색
## (패키지명: libgeographic-dev / libGeographic.so)
## -----------------------------------------------------------------------------
find_path(GEOGRAPHIC_INCLUDE_DIR GeographicLib/LocalCartesian.hpp)
find_library(GEOGRAPHIC_LIBRARY NAMES Geographic GeographicLib)

if(NOT GEOGRAPHIC_INCLUDE_DIR OR NOT GEOGRAPHIC_LIBRARY)
  message(FATAL_ERROR
    "Geographic(lib) not found. Install it first:\n"
    "  sudo apt-get update && sudo apt-get install -y libgeographic-dev")
endif()

## -----------------------------------------------------------------------------
## Catkin package
## -----------------------------------------------------------------------------
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GEOGRAPHIC_INCLUDE_DIR}   # ← 헤더 경로(예: /usr/include)
)

## -----------------------------------------------------------------------------
## Executables
## -----------------------------------------------------------------------------
# add_executable(ego_path_recorder_bag_kante src/ego_path_recorder_bag_kante.cpp)
# target_compile_features(ego_path_recorder_bag_kante PRIVATE cxx_std_11)
# target_link_libraries(ego_path_recorder_bag_kante ${catkin_LIBRARIES})

# add_executable(ego_path_recorder_csv_kante src/ego_path_recorder_csv_kante.cpp)
# target_compile_features(ego_path_recorder_csv_kante PRIVATE cxx_std_11)
# target_link_libraries(ego_path_recorder_csv_kante ${catkin_LIBRARIES})

# NOTE: Most source files are missing and need to be imported from src_old/
# Only building files that exist in src/
# add_executable(ego_path_recorder src/ego_path_recorder.cpp)
# add_executable(global_path_pub_kante src/global_path_pub_kante.cpp)
# add_executable(local_path_pub_kante src/local_path_pub_kante.cpp)
# add_executable(pp_1026 src/control/pp_1026.cpp)
# add_executable(pp_1026_noclass_test src/control/pp_1026_noclass_test.cpp)
# add_executable(path_visualizer src/path_visualizer.cpp)
# add_executable(pp_1102 src/decision/pp_1102.cpp)
# add_executable(pp_local_path_1102 src/pp_local_path_1102.cpp)
# add_executable(pp_fixed_speed src/decision/pp_fixed_speed.cpp)
# add_executable(gap_navigator_morai src/decision/gap_navigator_morai.cpp)
# add_executable(stanely_1105 src/control/stanely_1105.cpp)
# add_executable(curvature_control_ver2 src/control/curvature_control_ver2.cpp)
# add_executable(curvature_control_verPID src/control/curvature_control_verPID.cpp)
# add_executable(navigator_avoidance_with_lidar src/decision/navigator_avoidance_with_lidar.cpp)

# ======================================
# AVAILABLE TARGETS (files that exist)
# ======================================
# add_executable(convert_lidar_kante src/perception/convert_lidar_kante.cpp)
# target_link_libraries(convert_lidar_kante ${catkin_LIBRARIES} ${GEOGRAPHIC_LIBRARY})
# add_dependencies(convert_lidar_kante ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# add_executable(obstacle_detector src/perception/obstace_detector.cpp)
# target_link_libraries(obstacle_detector ${catkin_LIBRARIES} ${GEOGRAPHIC_LIBRARY})

# ----------------------------------------
# Path Recorder (GPS to CSV in ENU coordinates using GeographicLib)
add_executable(path_recorder src/control/gps_enu_path_recorder.cpp)
target_compile_features(path_recorder PRIVATE cxx_std_11)
target_link_libraries(path_recorder ${catkin_LIBRARIES} ${GEOGRAPHIC_LIBRARY})
add_dependencies(path_recorder ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ----------------------------------------
# Hybrid controller (refactored with reference structure)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP yaml-cpp)
add_executable(hybrid src/control/hybrid_control_node.cpp src/control/hybrid_function.cpp)
target_include_directories(hybrid PRIVATE src/control)
target_compile_features(hybrid PRIVATE cxx_std_11)
target_link_libraries(hybrid ${catkin_LIBRARIES} ${GEOGRAPHIC_LIBRARY} ${YAML_CPP_LIBRARIES})
add_dependencies(hybrid ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ----------------------------------------
# LIDAR Cluster Car Detection
find_package(PCL REQUIRED)
add_executable(cluster_car_detection src/perception/lidar/cluster_car_detection.cpp)
target_compile_features(cluster_car_detection PRIVATE cxx_std_11)
target_link_libraries(cluster_car_detection ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_include_directories(cluster_car_detection PRIVATE ${PCL_INCLUDE_DIRS})
add_dependencies(cluster_car_detection ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ----------------------------------------
# LIDAR AABB Costmap Node (with RANSAC ground filtering and clustering)
add_executable(lidar_aabb_costmap_node src/perception/lidar/lidar_aabb_costmap_node.cpp)
target_compile_features(lidar_aabb_costmap_node PRIVATE cxx_std_11)
target_link_libraries(lidar_aabb_costmap_node ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_include_directories(lidar_aabb_costmap_node PRIVATE ${PCL_INCLUDE_DIRS})
add_dependencies(lidar_aabb_costmap_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ----------------------------------------
# LIDAR AABB Costmap Node Ver2 (alternative version)
add_executable(lidar_aabb_ver2 src/perception/lidar/lidar_aabb_ver2.cpp)
target_compile_features(lidar_aabb_ver2 PRIVATE cxx_std_11)
target_link_libraries(lidar_aabb_ver2 ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_include_directories(lidar_aabb_ver2 PRIVATE ${PCL_INCLUDE_DIRS})
add_dependencies(lidar_aabb_ver2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ----------------------------------------
# LIDAR Patch Test Node
add_executable(patch src/perception/lidar/patch.cpp)
target_compile_features(patch PRIVATE cxx_std_11)
target_link_libraries(patch ${catkin_LIBRARIES} ${PCL_LIBRARIES})
target_include_directories(patch PRIVATE ${PCL_INCLUDE_DIRS})
add_dependencies(patch ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})