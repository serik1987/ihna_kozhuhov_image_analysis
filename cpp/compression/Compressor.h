//
// Created by serik1987 on 06.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSOR_H

#include "BaseCompressor.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Provides the file compression
     */
    class Compressor: public BaseCompressor {
    private:
        void writeSoftChunk(const SoftChunk* psoft, std::ofstream& output);
        void writeCompChunk(std::ofstream& output);

    protected:
        std::string getOutputFile(const std::string& input_file) override;
        void writeHeader(TrainSourceFile& file, std::ofstream& out) override;
        void writeConsequtiveFrame(std::ifstream& in, std::ofstream& out) override;

    public:
        /**
         * Initializes the compressor
         *
         * @param train the native file train (instance of StreamFileTrain). The train is assumed to be opened before
         * the compression process
         * @param output_path Folder where all compressed files shall be written
         */
        explicit Compressor(StreamFileTrain& train, const std::string& output_path):
            BaseCompressor(train, output_path) {};

        Compressor(const Compressor& other) = delete;
        Compressor& operator=(const Compressor& other) = delete;
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSOR_H
