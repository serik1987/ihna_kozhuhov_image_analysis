//
// Created by serik1987 on 18.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAIN_H

#include "../../init.h"

#include "FileTrain.h"
#include "CompressedSourceFile.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Train that contains compressed files only
     *
     * A file train is a sequence of files (e.g., T_1BF.0A00, T_1BF.0A01,T_1BF.0A02) that contains
     * a single record
     */
    class CompressedFileTrain : public FileTrain {
    protected:
        TrainSourceFile* createFile(const std::string& path, const std::string& filename,
                                    TrainSourceFile::NotInHead notInHead, const std::string& trainName) override{
            return new CompressedSourceFile(path, filename, notInHead, trainName);
        }

        /**
         *
         * @return the desired size of the ISOI chunk
         */
        uint32_t getDesiredIsoiChunkSize(TrainSourceFile& file) override;

        /**
         *
         * @return difference between the desired and the actual file size
         */
        uint32_t getFileSizeChecksum(TrainSourceFile& file) override { return 0; };

    public:
        CompressedFileTrain(const std::string& path, const std::string& filename, bool traverse):
            FileTrain(path, filename, traverse) {};

        class comp_chunk_not_exist_exception: public SourceFile::source_file_exception{
        public:
            explicit comp_chunk_not_exist_exception(SourceFile* file):
                    SourceFile::source_file_exception(MSG_COMP_CHUNK_NOT_EXIST_EXCEPTION, file) {};
            explicit comp_chunk_not_exist_exception(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_COMP_CHUNK_NOT_EXIST_EXCEPTION,
                                filename, trainname) {};
        };

        class compressed_frame_read_exception: public train_exception{
        public:
            explicit compressed_frame_read_exception(FileTrain* train):
                train_exception(MSG_COMPRESSED_FRAME_READ_ERROR, train) {};
            explicit compressed_frame_read_exception(const std::string& name):
                train_exception(MSG_COMPRESSED_FRAME_READ_ERROR, name) {};
        };

        /**
         * Moves the pointer to an appropriate frame
         *
         * @param n frame number
         * @result reference to the TrainSourceFile that contains an appropriate frame
         */
        TrainSourceFile& seek(int n) override;
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAIN_H
