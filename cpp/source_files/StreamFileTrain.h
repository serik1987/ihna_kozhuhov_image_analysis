//
// Created by serik1987 on 18.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAIN_H

#include "FileTrain.h"
#include "StreamSourceFile.h"

namespace iman {

    /**
     * A base class for all file trains.
     *
     * A file train is a sequence of files (e.g., T_1BF.0A00, T_1BF.0A01,T_1BF.0A02) that contains
     * a single record
     *
     * Stream file train represents a train that contains original (uncompressed) data
     */
    class StreamFileTrain: public FileTrain {
    private:
        int file_idx = 0;
        const uint32_t* file_sizes = nullptr;

    protected:
        TrainSourceFile* createFile(const std::string& path, const std::string& filename,
                                            TrainSourceFile::NotInHead notInHead) override {
            return new StreamSourceFile(path, filename, notInHead);
        };

        uint32_t getDesiredIsoiChunkSize(TrainSourceFile& file) override;
        uint32_t getFileSizeChecksum(TrainSourceFile& file) override;

    public:
        /**
         * Creates new file train
         *
         * @param path full path to the file train
         * @param filename filename (without any extension)
         * @param sizes array containing the actual file sizes as this is defined by the operating system
         * @param traverse uses in case when filename doesn't refer to the head of the file
         * true - the file will be substituted to another file that is in the file head
         * false - TrainSourceFile::not_train_head will be thrown
         */
        StreamFileTrain(const std::string& path, const std::string& filename, const uint32_t sizes[],
                bool traverse = false):
            FileTrain(path, filename, traverse), file_sizes(sizes) {};

        void open() override {
            file_idx = 0;
            FileTrain::open();
        }
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAIN_H
