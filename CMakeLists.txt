cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(SamCPP)
find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)

set(onnxruntime_dir  C:/OGC/onnxruntime-win-x64-gpu )

add_executable(sam_cpp_test   test.cpp sam.cpp)

if (APPLE)
  set(onnxruntime_lib ${onnxruntime_dir}/lib/onnxruntime.dylib)
else()
  set(onnxruntime_lib ${onnxruntime_dir}/lib/onnxruntime.lib)
endif()
target_include_directories(
  sam_cpp_test PUBLIC 
  ${onnxruntime_dir}/include
  ${OpenCV_INCLUDE_DIRS}/include
)
target_link_libraries(
  sam_cpp_test PRIVATE
  ${onnxruntime_lib}
  ${OpenCV_LIBS}
  gflags
)
