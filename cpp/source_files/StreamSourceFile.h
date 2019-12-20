//
// Created by serik1987 on 17.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMSOURCEFILE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMSOURCEFILE_H

#include "TrainSourceFile.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Represents a regular file where imaging data were stored in their original
     * decompressed way
     */
    class StreamSourceFile: public TrainSourceFile {
    public:
        /**
         * Creates new train source file
         *
         * @param path full path to the file
         * @param filename the file name
         * @param notInHead option that will be applied during the loadFileInfo
         * The option will not affect when you load information about the head file.
         * However, when the file mentioned in the argument is not the head file
         * the behaviour is the following:
         * NotInHeadIgnore - don't care about this. This is suitable when the file train
         * (see help on FileTrain) has loaded at least one file and tries to add new
         * files at the end of the train
         * NotInHeadFail - will throw an exception. This option is suitable when you try
         * to make the list of all available data in the Python. In this case all not-in-head
         * files shall behave like any files unrelated to IMAN.
         * NotInHeadTraverse - will traverse to the beginning of the file. This option
         * is suitable when the file train (FileTrain) tries to add its first file and assumes
         * that this is the beginning of the train
         * @param trainName - parameter that influence on the text of the error messages only
         */
        StreamSourceFile(const std::string& path, const std::string& filename, NotInHead notInHead = NotInHeadFail,
                const std::string& trainName = ""):
            TrainSourceFile(path, filename, notInHead, trainName) {};

        class not_stream_file: public source_file_exception{
        public:
            explicit not_stream_file(StreamSourceFile* file):
                source_file_exception(MSG_NOT_STREAM_FILE, file) {};
            explicit not_stream_file(const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_NOT_STREAM_FILE, filename, trainname) {};
        };

        /**
         * Loads the file information
         */
        void loadFileInfo() override;

        [[nodiscard]] std::string getFileTypeDescription() const override { return "uncompressed imaging data"; }
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMSOURCEFILE_H
