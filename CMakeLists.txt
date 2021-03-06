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
        cpp/source_files/CompressedFileTrain.cpp cpp/source_files/Frame.cpp cpp/compression/BaseCompressor.cpp
        cpp/compression/Compressor.cpp cpp/compression/Decompressor.cpp cpp/tracereading/PixelListItem.cpp
        cpp/tracereading/TraceReader.cpp cpp/synchronization/Synchronization.cpp
        cpp/synchronization/NoSynchronization.cpp cpp/synchronization/QuasiStimulusSynchronization.cpp
        cpp/synchronization/QuasiTimeSynchronization.cpp cpp/synchronization/ExternalSynchronization.cpp
        cpp/isolines/Isoline.cpp cpp/isolines/NoIsoline.cpp cpp/isolines/LinearFitIsoline.cpp
        cpp/isolines/TimeAverageIsoline.cpp cpp/tracereading/TraceReaderAndCleaner.cpp cpp/misc/LinearFit.cpp
        cpp/accumulators/Accumulator.cpp cpp/accumulators/TraceAutoReader.cpp cpp/accumulators/FrameAccumulator.cpp
        cpp/accumulators/MapPlotter.cpp cpp/accumulators/MapFilter.cpp)

include_directories(/usr/include/python3.6m)

add_library(imageanalysis SHARED imageanalysis.cpp cpp/source_files/SourceFile.cpp cpp/source_files/ChunkHeader.cpp
        cpp/source_files/Chunk.cpp cpp/source_files/FramChunk.cpp cpp/source_files/FramCostChunk.cpp
        cpp/source_files/FramEpstChunk.cpp cpp/source_files/IsoiChunk.cpp cpp/source_files/HardChunk.cpp
        cpp/source_files/SoftChunk.cpp cpp/source_files/CostChunk.cpp cpp/source_files/EpstChunk.cpp
        cpp/source_files/GreenChunk.cpp cpp/source_files/DataChunk.cpp cpp/source_files/SyncChunk.cpp
        cpp/source_files/RoisChunk.cpp cpp/source_files/CompChunk.cpp cpp/source_files/AnalysisSourceFile.cpp
        cpp/source_files/GreenSourceFile.cpp cpp/source_files/TrainSourceFile.cpp cpp/source_files/StreamSourceFile.cpp
        cpp/source_files/CompressedSourceFile.cpp cpp/source_files/FileTrain.cpp cpp/source_files/StreamFileTrain.cpp
        cpp/source_files/CompressedFileTrain.cpp cpp/source_files/Frame.cpp cpp/compression/BaseCompressor.cpp
        cpp/compression/Compressor.cpp cpp/compression/Decompressor.cpp cpp/tracereading/PixelListItem.cpp
        cpp/tracereading/TraceReader.cpp cpp/synchronization/Synchronization.cpp
        cpp/synchronization/NoSynchronization.cpp cpp/synchronization/QuasiStimulusSynchronization.cpp
        cpp/synchronization/QuasiTimeSynchronization.cpp cpp/synchronization/ExternalSynchronization.cpp
        cpp/isolines/Isoline.cpp cpp/isolines/NoIsoline.cpp cpp/isolines/LinearFitIsoline.cpp
        cpp/isolines/TimeAverageIsoline.cpp cpp/tracereading/TraceReaderAndCleaner.cpp cpp/misc/LinearFit.cpp
        cpp/accumulators/Accumulator.cpp cpp/accumulators/TraceAutoReader.cpp cpp/accumulators/FrameAccumulator.cpp
        cpp/accumulators/MapPlotter.cpp cpp/accumulators/MapFilter.cpp)