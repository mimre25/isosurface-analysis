cmake_minimum_required(VERSION 3.2)


project(isosurfaces_lib)
MESSAGE("===============")
MESSAGE("ISOSURFACES_LIB")


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)


find_package(SDL2 REQUIRED)
set(THOR_LIB /usr/local/lib/libThorSerialize17.so)
set(GCC_FOR_CUDA gcc-5)
set(LINKER_FLAGS "-lboost_program_options -lboost_regex -lSDL2 -lSDL2main -lboost_system -lboost_filesystem -lcuda -lcudart")


set(ADDITIONAL_FLAGS "-g -Wno-error=switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 ${LINKER_FLAGS} ${ADDITIONAL_FLAGS}")#-Wno-narrowing



find_package(CUDA REQUIRED)


set(SOURCE_FILES
    types/src/MarchingCubes.cpp types/hdr/MarchingCubes.h
    types/hdr/MarchingCubes.h runtime/hdr/globals.h types/src/Isosurface.cpp types/hdr/Isosurface.h
    types/hdr/Isosurface.h filehandler/src/FileWriter.cpp filehandler/hdr/FileWriter.h
    filehandler/hdr/FileWriter.h filehandler/src/FileReader.cpp filehandler/hdr/FileReader.h
    filehandler/hdr/FileReader.h filehandler/src/FileHandler.cpp filehandler/hdr/FileHandler.h
    filehandler/hdr/FileHandler.h utils/src/StringUtils.cpp utils/hdr/StringUtils.h utils/hdr/StringUtils.h
    types/src/DistanceField.cpp types/hdr/DistanceField.h types/hdr/DistanceField.h filehandler/src/BinaryFloatReader.cpp filehandler/hdr/BinaryFloatReader.h
    filehandler/hdr/BinaryFloatReader.h filehandler/src/FloatBinaryWriter.cpp filehandler/hdr/FloatBinaryWriter.h
    filehandler/hdr/FloatBinaryWriter.h types/src/JointHistogram.cpp types/hdr/JointHistogram.h types/src/JointHistogram.cpp types/hdr/JointHistogram.h
    visualization/src/HistogramVisualizer.cpp visualization/hdr/HistogramVisualizer.h visualization/hdr/HistogramVisualizer.h types/src/SimilarityMap.cpp types/hdr/SimilarityMap.h types/hdr/SimilarityMap.h
    visualization/src/SimilarityMapVisualizer.cpp visualization/hdr/SimilarityMapVisualizer.h visualization/hdr/SimilarityMapVisualizer.h
    types/src/Settings.cpp types/hdr/Settings.h types/hdr/Settings.h
    utils/src/Report.cpp utils/hdr/Report.h utils/hdr/Report.h utils/src/VectorUtils.cpp utils/hdr/VectorUtils.h
    runtime/src/globals.cpp runtime/hdr/globals.h runtime/src/MultiVar.cpp runtime/hdr/MultiVar.h runtime/src/Single.cpp runtime/hdr/Single.h utils/src/MemUtils.cpp utils/hdr/MemUtils.h
    json/src/JSONHandler.cpp json/hdr/JSONHandler.h json/src/JSONObject.cpp json/hdr/JSONObject.h
    DAO/src/RunInformation.cpp DAO/hdr/RunInformation.h DAO/src/RepresentativesInfo.cpp DAO/hdr/RepresentativesInfo.h DAO/src/SimilarityMapInformation.cpp DAO/hdr/SimilarityMapInformation.h DAO/src/VolumeInformation.cpp DAO/hdr/VolumeInformation.h
    types/src/VariableNameAnalyzer.cpp types/hdr/VariableNameAnalyzer.h runtime/src/PrintOnly.cpp runtime/hdr/PrintOnly.h
    utils/src/floatNExtensions.cpp utils/hdr/floatNExtensions.h
    runtime/src/Scripting.cpp runtime/hdr/Scripting.h
    DAO/src/DataInfo.cpp DAO/hdr/DataInfo.h DAO/src/ScriptingInput.cpp DAO/hdr/ScriptingInput.h
    DAO/src/SingleConfig.cpp DAO/hdr/SingleConfig.h
    DAO/src/MultiVarConfig.cpp DAO/hdr/MultiVarConfig.h
    DAO/src/VariableInfo.cpp DAO/hdr/VariableInfo.h
    isosurfaces_cuda/funcs.cpp isosurfaces_cuda/funcs.h
    runtime/src/SurfaceToSurface.cpp runtime/hdr/SurfaceToSurface.h
    runtime/src/DistanceEqualization.cpp runtime/hdr/DistanceEqualization.h
    utils/src/FileUtils.cpp utils/hdr/FileUtils.h
    utils/hdr/FileUtils.h utils/hdr/disablePrint.h runtime/src/CreateMapOnly.cpp runtime/hdr/CreateMapOnly.h runtime/hdr/CreateMapOnly.h runtime/src/Stitch.cpp runtime/hdr/Stitch.h runtime/src/Check.cpp runtime/hdr/Check.h utils/src/RuntimeInfoUtils.cpp utils/hdr/RuntimeInfoUtils.h
    utils/typeOperation.h isosurfaces_cuda/hdr/helper_math.h isosurfaces_cuda/src/defines.h isosurfaces_cuda/src/tables.h runtime/src/CheckMap.cpp runtime/hdr/CheckMap.h runtime/src/Histogram.cpp runtime/hdr/Histogram.h)

set(CUDA_FILES isosurfaces_cuda/src/common.cu isosurfaces_cuda/hdr/helpers.h
    isosurfaces_cuda/src/DistanceField.cu isosurfaces_cuda/src/IsosurfaceApproximation.cu
    isosurfaces_cuda/src/JointHistogram.cu isosurfaces_cuda/src/BVH.cu isosurfaces_cuda/src/thrustWrapper.cu
    isosurfaces_cuda/src/IsosurfaceExtrema.cu
    isosurfaces_cuda/src/MarchingCube.cu isosurfaces_cuda/src/MarchingCube.cpp)

set(PROJECT_DIRS
    ${CMAKE_SOURCE_DIR}
    )



find_package(Boost COMPONENTS program_options filesystem system regex REQUIRED)



set(CUDA_ADDITIONAL_FLAGS "-ccbin ${GCC_FOR_CUDA} --Wno-deprecated-gpu-targets")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_ADDITIONAL_FLAGS})
#set(CUDA_NVCC_FLAGS -g ${CUDA_ADDITIONAL_FLAGS}) #add debug only for host device code
#set(CUDA_NVCC_FLAGS -G ${CUDA_ADDITIONAL_FLAGS}) #add debug for host & device code




include_directories(${THOR_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

cuda_include_directories(${SDL2_INCLUDE_DIR} ${PROJECT_DIRS} ${Boost_INCLUDE_DIRS})

cuda_add_library(isosurfaces_cuda ${CUDA_FILES})

add_library(isosurfaces_lib ${SOURCE_FILES})
target_include_directories(isosurfaces_lib PUBLIC ${SDL2_INCLUDE_DIR} ${PROJECT_DIRS} ${Boost_INCLUDE_DIRS} ${THOR_INCLUDE_DIRS})
#set(LINKER_FLAGS "-lboost_program_options -lboost_regex -lSDL2 -lSDL2main -lboost_system -lboost_filesystem -static-libgcc -static-stdlibc++")

target_link_libraries(isosurfaces_lib
    isosurfaces_cuda
    ${Boost_LIBRARIES}
    ${SDL2_LIBRARY}
    ${THOR_LIB})

MESSAGE("ISOSURFACES_LIB DONE")
MESSAGE("================")
