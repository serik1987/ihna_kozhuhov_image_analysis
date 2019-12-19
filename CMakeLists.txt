cmake_minimum_required(VERSION 3.15)
project(ihna_kozhukhov_image_analysis)

set(CMAKE_CXX_STANDARD 17)

add_executable(test main.cpp cpp/source_files/SourceFile.cpp cpp/source_files/ChunkHeader.cpp
        cpp/source_files/Chunk.cpp cpp/source_files/FramChunk.cpp cpp/source_files/FramCostChunk.cpp
        cpp/source_files/FramEpstChunk.cpp cpp/source_files/IsoiChunk.cpp cpp/source_files/HardChunk.cpp
        cpp/source_files/SoftChunk.cpp cpp/source_files/CostChunk.cpp cpp/source_files/EpstChunk.cpp
        cpp/source_files/GreenChunk.cpp cpp/source_files/DataChunk.cpp cpp/source_files/SyncChunk.cpp
        cpp/source_files/RoisChunk.cpp cpp/source_files/CompChunk.cpp cpp/source_files/AnalysisSourceFile.cpp
        cpp/source_files/GreenSourceFile.cpp cpp/source_files/TrainSourceFile.cpp cpp/source_files/StreamSourceFile.cpp
        cpp/source_files/CompressedSourceFile.cpp cpp/source_files/FileTrain.cpp cpp/source_files/StreamFileTrain.cpp
        cpp/source_files/CompressedFileTrain.cpp)

include_directories(/usr/include/python3.6m)

add_library(exceptions SHARED c_python/source_files/exceptions.cpp)