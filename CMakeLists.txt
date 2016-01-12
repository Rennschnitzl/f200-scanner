project(f200-scanner)
cmake_minimum_required(VERSION 2.8)

if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x") #-Wall
endif()

aux_source_directory(. SRC_LIST)
aux_source_directory(./realsense-core RSC_SRC_LIST)
find_package(OpenCV 2.4.9 REQUIRED )
find_package(PCL REQUIRED )
#find_package(octomap REQUIRED)
#find_package(aruco REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ) #${OCTOMAP_INCLUDE_DIRS}
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(${PROJECT_NAME} ${SRC_LIST} ${RSC_SRC_LIST})
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} ${PCL_LIBRARIES}) #${OCTOMAP_LIBRARIES} ${aruco_LIBS}
