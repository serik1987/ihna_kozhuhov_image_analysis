//
// Created by serik1987 on 06.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DECOMPRESSOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DECOMPRESSOR_H

#include "BaseCompressor.h"
#include "../source_files/CompChunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Provides decompression of the compressed file
     */
    class Decompressor: public BaseCompressor {
    private:
        void writeSoftChunk(const SoftChunk& input_chunk, std::ofstream& output);
        void checkCompChunk(const CompChunk& input_chunk);

    protected:
        std::string getOutputFile(const std::string& input_file) override;
        void writeHeader(TrainSourceFile& input, std::ofstream& output) override;
        void writeConsequtiveFrame(std::ifstream& in, std::ofstream& out) override;

    public:
        /**
         * Initializes the decompression
         *
         * @param train file train to decompress. The train is assumed to be opened
         * @param output_folder Folder where the decompressed data shall be written
         */
        explicit Decompressor(CompressedFileTrain& train, const std::string& output_folder):
            BaseCompressor(train, output_folder) {};
        Decompressor(const Decompressor& other) = delete;
        Decompressor& operator=(const Decompressor& other) = delete;

        class decompression_exception: public iman_exception{
        public:
            decompression_exception(): iman_exception(MSG_DECOMPRESSION_EXCEPTION) {};
        };

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DECOMPRESSOR_H
