cmake_minimum_required(VERSION 3.15)
project(ihna_kozhuhov_image_analysis)

set(CMAKE_CXX_STANDARD 17)

add_executable(ihna_kozhuhov_image_analysis main.cpp cpp/source_files/SourceFile.cpp cpp/source_files/ChunkHeader.cpp)