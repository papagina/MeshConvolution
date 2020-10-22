cmake_minimum_required(VERSION 3.5)
project(GraphSampling)

set(CMAKE_CXX_STANDARD 14)
find_package(ZLIB)

find_package( OpenCV REQUIRED )

add_executable(GraphSampling main.cpp mdSArray.h mdVector.h mdVector.inl meshLoader.h meshPooler.h meshPooler_visualizer_new.h meshCNN.h cnpy/cnpy.cpp cnpy/cnpy.h)

target_link_libraries(GraphSampling ${ZLIB_LIBRARIES})

target_link_libraries(GraphSampling ${OPENCV_LIBRARIES})
