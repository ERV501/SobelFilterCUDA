cmake_minimum_required(VERSION 2.8)
project( PracticaCUDA C CUDA)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets -O3 -std=c++11 -Xcompiler -fopenmp")
add_executable( Sobel_Filter Image_filter.cu)
target_link_libraries( Sobel_Filter ${OpenCV_LIBS} )
