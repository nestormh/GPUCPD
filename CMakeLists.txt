cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(gpucpd)

# set(OpenCV_DIR "/usr/local/share/OpenCV")
# set(PCL_DIR "/usr/local/share/pcl-1.6")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -ffast-math -march=native")

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
# find_package(OMPL    REQUIRED )
set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.49.0 COMPONENTS filesystem system program_options)
find_package(CUDA 5.0 REQUIRED)
find_package(OpenCV  REQUIRED )
# find_package( Eigen3    REQUIRED )
# find_package(PCL REQUIRED)
# find_package(PCL REQUIRED)

set (EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
# add_subdirectory(tutorials)
add_subdirectory(src)