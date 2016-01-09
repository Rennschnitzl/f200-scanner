project(f200-scanner)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
aux_source_directory(./realsense-core RSC_SRC_LIST)
find_package(OpenCV 2.4.9 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(${PROJECT_NAME} ${SRC_LIST} ${RSC_SRC_LIST})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )

