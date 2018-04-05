add_executable(tensorrt main.cpp trtengine.cpp common.cpp)
add_dependencies(tensorrt igvc_msgs_gencpp)
target_link_libraries(tensorrt ${catkin_LIBRARIES} ${PCL_LIBRARIES})
