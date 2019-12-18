//
// Created by serik1987 on 18.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAIN_H

#include <list>
#include "TrainSourceFile.h"

namespace iman {

    /**
     * A base class for all file trains.
     *
     * A file train is a sequence of files (e.g., T_1BF.0A00, T_1BF.0A01,T_1BF.0A02) that contains
     * a single record
     */
    class FileTrain: private std::list<TrainSourceFile*> {
    private:
        std::string path = "";
        std::string filename = "";
        bool traverse = false;
        bool opened = false;
        uint32_t fileHeaderSize = -1;
        uint32_t frameHeaderSize = -1;

    protected:
        /**
         *
         * @return true if the data is needed to be traversed to forward in case if there is no head
         * false if exception shall be thrown in this case
         */
        [[nodiscard]] bool isTraverse() const { return traverse; }

        /**
         * Creates an instance for an arbitrary train source file. Doesn't open it. Doesn't traverse it
         * The method shall not return nullptr, otherwise the function may be failed. Please, throw an exception
         * when something goes wrong but don't return nullptr
         *
         * @param path the same parameters and results as those needed for TrainSourceFile constructor
         * @param filename
         * @param notInHead
         * @return
         */
        virtual TrainSourceFile* createFile(const std::string& path, const std::string& filename,
                TrainSourceFile::NotInHead notInHead) = 0;

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
        FileTrain(const std::string& path, const std::string& filename, bool traverse = false){
            this->path = path;
            this->filename = filename;
            this->traverse = traverse;
        }

        virtual ~FileTrain();

        /**
         *
         * @return total number of files opened
         */
        [[nodiscard]] size_t getFileNumber() const { return size(); }

        /**
         *
         * @return full path to the file train
         */
        [[nodiscard]] const std::string& getFilePath() const { return path; }

        /**
         *
         * @return filename for the train head file
         */
        [[nodiscard]] const std::string& getFilename() const { return filename; }

        /**
         *
         * @return size of frame header in bytes or uint32_t(-1) if the train is not opened
         */
        [[nodiscard]] uint32_t getFrameHeaderSize() const { return frameHeaderSize; }

        /**
         *
         * @return size of each file header or meaningless value if the train is not opened
         */
        [[nodiscard]] uint32_t getFileHeaderSize() const { return fileHeaderSize; }

        /**
         *
         * @return true if the train has been opened
         */
        [[nodiscard]] bool isOpened() const { return opened; }

        /**
         * Prints general information about the file train
         *
         * @param out output stream where such information shall be printed (reference to it)
         * @param train reference to the train
         * @return reference to the output stream
         */
        friend std::ostream& operator<<(std::ostream& out, const FileTrain& train);

        /**
         *
         * @return iterator to the beginning of the train
         */
        [[nodiscard]] auto begin() { return std::list<TrainSourceFile*>::begin(); }

        /**
         *
         * @return iterator to the end of the train
         */
        [[nodiscard]] auto end() { return std::list<TrainSourceFile*>::end(); }

        /**
         * Opens all files within the file train
         */
        void open();

        /**
         * Closes all files within the train
         */
        void close();
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAIN_H
