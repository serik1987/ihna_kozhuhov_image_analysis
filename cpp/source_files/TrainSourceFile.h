//
// Created by serik1987 on 17.12.2019.
//

#ifndef IHNA_KOZKUKHOV_IMAGE_ANALYSIS_TRAINSOURCEFILE_H
#define IHNA_KOZKUKHOV_IMAGE_ANALYSIS_TRAINSOURCEFILE_H

#include "SourceFile.h"

namespace iman {

    /**
     * Base class for so called "train files".
     * The "train" is a sequence of files representing native or compressed
     * optical imaging signal. Each file represent only part of the signal.
     *
     * The file that represents the beginning of the signal is called "head file"
     * The file that represents the end of the signal is called "tail file"
     * Any file representing a part of the signal refers to the file that represents
     * the previous part and to the file that represents the next part. Such
     * references are placed into SOFT chunk.
     */
    class TrainSourceFile: public SourceFile {
    public:
        enum NotInHead {NotInHeadIgnore, NotInHeadFail, NotInHeadTraverse};

    private:
        NotInHead loadFileInfoMode;

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
         */
        TrainSourceFile(const std::string& path, const std::string& filename, NotInHead notInHead = NotInHeadFail):
            SourceFile(path, filename), loadFileInfoMode(notInHead) {};

        class not_train_head: public source_file_exception{
        public:
            explicit not_train_head(TrainSourceFile* file):
                source_file_exception("The file is not the first file in the train", file) {};
        };

        void loadFileInfo() override;

        /**
         *
         * @return description of the desired file type
         */
        [[nodiscard]] std::string getFileTypeDescription() const override = 0;

    };

}


#endif //IHNA_KOZKUKHOV_IMAGE_ANALYSIS_TRAINSOURCEFILE_H
