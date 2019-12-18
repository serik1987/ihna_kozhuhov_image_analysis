//
// Created by serik1987 on 18.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAIN_H

#include "FileTrain.h"
#include "CompressedSourceFile.h"

namespace iman {

    /**
     * Train that contains compressed files only
     *
     * A file train is a sequence of files (e.g., T_1BF.0A00, T_1BF.0A01,T_1BF.0A02) that contains
     * a single record
     */
    class CompressedFileTrain : public FileTrain {
    protected:
        TrainSourceFile* createFile(const std::string& path, const std::string& filename,
                                    TrainSourceFile::NotInHead notInHead) override{
            return new CompressedSourceFile(path, filename, notInHead);
        }

    public:
        CompressedFileTrain(const std::string& path, const std::string& filename, bool traverse):
            FileTrain(path, filename, traverse) {};
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAIN_H
