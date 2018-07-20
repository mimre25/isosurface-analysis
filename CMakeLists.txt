cmake_minimum_required(VERSION 3.2)

set(CRC TRUE)

MESSAGE("================")
MESSAGE("ISOSURFACES")
MESSAGE("CRC: ${CRC}")
project(isosurfaces)

#SET (CMAKE_CXX_COMPILER "/opt/crc/g/gcc/6.2.0/bin/g++")

MESSAGE("CXX: ${CMAKE_CXX_COMPILER}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 ${LINKER_FLAGS} ${ADDITIONAL_FLAGS}")#-Wno-narrowing

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)




find_package(CUDA REQUIRED)

add_executable(isosurfaces isosurfaces_lib/runtime/main.cpp)

include_directories(isosurfaces isosurfaces_lib ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})



set(PROJECT_DIRS
		${CMAKE_SOURCE_DIR}
		)


add_subdirectory(isosurfaces_lib)

target_link_libraries(isosurfaces isosurfaces_lib)


MESSAGE("===============")
MESSAGE("ISOSURFACES DONE")

