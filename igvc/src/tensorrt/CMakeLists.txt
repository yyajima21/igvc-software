cuda_compile(tensorrt trtengine.cpp common.cpp )
cuda_add_executable(tensorrt main.cpp)
add_dependencies(tensorrt igvc_msgs_gencpp)
target_link_libraries(tensorrt -lnvinfer -lnvparsers -lnvinfer_plugin ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
