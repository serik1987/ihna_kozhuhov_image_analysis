//
// Created by serik1987 on 06.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_BASECOMPRESSOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_BASECOMPRESSOR_H

#include "../exceptions.h"
#include "../source_files/StreamFileTrain.h"
#include "../source_files/CompressedFileTrain.h"

namespace GLOBAL_NAMESPACE {

    class BaseCompressor {
    protected:
        typedef uint16_t OriginalElement;
        typedef uint8_t CompressedElement;

        size_t original_frame_size = 0;
        size_t elements_in_frame = 0;

        OriginalElement* first_frame = nullptr;
        OriginalElement* current_frame = nullptr;
        CompressedElement* compressed_frame = nullptr;

        FileTrain& train;
        std::string output_path;

        /**
         * Returns the output file name.
         * Does not opens the file
         *
         * @param input_file the input file name
         * @return the output file name
         */
        virtual std::string getOutputFile(const std::string& input_file) =  0;

    public:
        /**
         * Initializes the compressor
         *
         * @param input The input file train. The train is assumed to be open()'ed
         * @param output_path the folder where all processed files shall be put. The folder name shall be end by
         * '/' in Linux or '\' in Windows
         */
        explicit BaseCompressor(FileTrain& input, const std::string& output_path);

        BaseCompressor(const BaseCompressor& other) = delete;
        BaseCompressor* operator=(const BaseCompressor& other) = delete;

        virtual ~BaseCompressor();

        friend std::ostream& operator<<(std::ostream& out, const BaseCompressor& compressor);

        /**
         * In case of compression, runs the compression process
         * In case of decompression, runs the decompression process
         */
        void run();
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_BASECOMPRESSOR_H
