pure pursuit + local path(with lidar)
1.convert_lidar_kante.cpp -> 뒤집힌 lidar를 전처리하여 /scan 토픽을 생성
2.pp_1102 -> pure pursuit으로 gps로 찍은 global path 를 주행하다가 , 장애물이 roi에 들어여면 doAvoid을 하게됨,
# --------------------------------------------------------------------------------
# cmake 참고_ 

# ver1 -> ONLY PRUE PURSUIT - KANTE UPLODADED / 1026 ~ 1031
add_executable(pp_1026 src/pp_1026.cpp)
target_compile_features(pp_1026 PRIVATE cxx_std_11)
# Geographic(lib) 는 이 타깃(pp_1026)에서만 필요
target_link_libraries(pp_1026 ${catkin_LIBRARIES} ${GEOGRAPHIC_LIBRARY})

# ver2 -> PURE PURSUIT + OBSTACLE_AVOIDANCE _ KANTE UPLODADED / 1102 ~
# pp_with_obstacle.launch로 실행이 가능하다.! 
add_executable(convert_lidar_kante src/convert_lidar_kante.cpp)
target_link_libraries(convert_lidar_kante ${catkin_LIBRARIES} Geographic)
add_dependencies(convert_lidar_kante ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(obstacle_detector src/perception/obstacle_detector.cpp)
target_link_libraries(obstacle_detector ${catkin_LIBRARIES} GeographicLib)

add_executable(obstacle_detector src/decision/obstacle_detector.cpp)
target_link_libraries(obstacle_detector ${catkin_LIBRARIES} GeographicLib)

----------------------------------------------------------------------------------------------------------
pure pursuit with MORAI

HZ문제
현재 구조(단일 스레드 + ros::spinOnce() + rate.sleep() )에선
spinOnce()에서 한 번 콜백들을 처리하고
제어 계산/퍼블리시 한 뒤
rate.sleep() 동안은 콜백이 돌지 않는다.
sleep 중 들어온 메시지는 구독 큐에 쌓였다가 다음 사이클의 spinOnce() 때 한꺼번에 처리된다.
 (단, 퍼블리셔 주파수가 높고 queue_size가 작으면 중간에 드롭될 수 있음)
