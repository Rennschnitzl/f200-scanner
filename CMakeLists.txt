project(f200-scanner)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
# this is needed for the findaruco.cmake script
SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )
find_package(OpenCV REQUIRED )
find_package(PCL REQUIRED )
find_package(aruco REQUIRED )
#find_package(octomap REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ) #${OCTOMAP_INCLUDE_DIRS}
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES} ${aruco_LIBS} realsense) #${OCTOMAP_LIBRARIES}
