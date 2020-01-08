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
    public:
        typedef void (*ProgressFunction)(float);

    private:
        void writeFirstFrame(std::istream& in, std::ostream& out);
        ProgressFunction progressFunction = nullptr;

    protected:
        typedef uint16_t OriginalElement;
        typedef uint8_t CompressedElement;
        typedef union {
            uint32_t L;
            uint8_t S[4];
        } EXTRA_PIXEL_RECORD;

        typedef union {
            uint16_t L;
            uint8_t S[2];
        } COMPRESSOR_RECORD;

        size_t original_frame_size = 0;
        size_t elements_in_frame = 0;
        uint32_t compressed_frame_number = 0;

        OriginalElement* previous_frame = nullptr;
        OriginalElement* current_frame = nullptr;
        CompressedElement* compressed_frame = nullptr;
        EXTRA_PIXEL_RECORD* extra_pixels = nullptr;
        uint32_t extra_pixels_number = 0;

        FileTrain& train;
        std::string output_path;
        std::string full_input_file;
        std::string full_output_file;

        /**
         * Returns the output file name.
         * Does not opens the file
         *
         * @param input_file the input file name
         * @return the output file name
         */
        virtual std::string getOutputFile(const std::string& input_file) =  0;

        /**
         * Writes the file header
         *
         * @param input input file (instance to TrainSourceFile
         * @param output output file
         */
        virtual void writeHeader(TrainSourceFile& input, std::ofstream& output) = 0;

        /**
         * Loads a single frame from the input stream, processes it (this may be compression or decompression)
         * and saves it to the output stream
         *
         * @param in the input stream
         * @param out the output stream
         */
        virtual void writeConsequtiveFrame(std::ifstream& in, std::ofstream& out) = 0;

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

        /**
         * Sets the function that updates the progress bar when the compression is in progress
         *
         * @param value pointer to the progress function that have the following type:
         * typedef void (*ProgressFunction)(float perc)
         * where perc is percentage completed
         */
        void setProgressFunction(ProgressFunction value) { progressFunction = value; }
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_BASECOMPRESSOR_H
