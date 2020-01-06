//
// Created by serik1987 on 06.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DECOMPRESSOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DECOMPRESSOR_H

#include "BaseCompressor.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Provides decompression of the compressed file
     */
    class Decompressor: public BaseCompressor {
    protected:
        std::string getOutputFile(const std::string& input_file) override;

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

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DECOMPRESSOR_H
