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
    protected:
        TrainSourceFile* createFile(const std::string& path, const std::string& filename,
                                            TrainSourceFile::NotInHead notInHead) override {
            return new StreamSourceFile(path, filename, notInHead);
        };

    public:
        /**
         * Creates new file train
         *
         * @param path full path to the file train
         * @param filename filename (without any extension)
         * @param traverse uses in case when filename doesn't refer to the head of the file
         * true - the file will be substituted to another file that is in the file head
         * false - TrainSourceFile::not_train_head will be thrown
         */
        StreamFileTrain(const std::string& path, const std::string& filename, bool traverse = false):
            FileTrain(path, filename, traverse) {};
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAIN_H
