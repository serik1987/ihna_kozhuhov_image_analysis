//
// Created by serik1987 on 18.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAIN_H

#include <vector>
#include <list>
#include "TrainSourceFile.h"

namespace GLOBAL_NAMESPACE {

    class Frame;

    /**
     * A base class for all file trains.
     *
     * A file train is a sequence of files (e.g., T_1BF.0A00, T_1BF.0A01,T_1BF.0A02) that contains
     * a single record
     */
    class FileTrain: private std::list<TrainSourceFile*> {
    public:
        enum ExperimentalMode {Continuous, Episodic, Unknown};

    private:
        std::string path = "";
        std::string filename = "";
        bool traverse = false;
        bool opened = false;
        uint32_t fileHeaderSize = -1;
        uint32_t frameHeaderSize = -1;
        ExperimentalMode experimentalMode = Unknown;
        int xSize = -1, ySize = -1;
        size_t xySize = -1, frameImageSize = -1, frameSize = -1;
        std::vector<int> synchChannelMax;
        int dataType = -1;
        int totalFrames = -1;

        /**
         * Loads all scalar train properties from the file
         *
         * @param file the train head file
         */
        void loadTrainProperties(TrainSourceFile& file);

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
         * @param path the same parameters and results as those needed for TrainSourceFile constructor. However,
         * no parameters are default
         * @param filename
         * @param notInHead
         * @param trainName
         * @return
         */
        virtual TrainSourceFile* createFile(const std::string& path, const std::string& filename,
                TrainSourceFile::NotInHead notInHead, const std::string& trainName) = 0;

        /**
         * Checks for train consistency
         */
        virtual void sanityCheck();

        /**
         *
         * @return the desired size of the ISOI chunk
         */
        virtual uint32_t getDesiredIsoiChunkSize(TrainSourceFile& file) = 0;

        /**
         *
         * @return difference between the desired and the actual file size
         */
        virtual uint32_t getFileSizeChecksum(TrainSourceFile& file) = 0;

    public:
        class train_exception: public io_exception{
        public:
            train_exception(const std::string& message, const FileTrain* train):
                io_exception(message, train->getFilePath() + train->getFilename()) {};
            train_exception(const std::string& message, const std::string& train_name):
                io_exception(message, train_name) {};
        };

        class experiment_mode_exception: public train_exception{
        public:
            explicit experiment_mode_exception(const FileTrain* train):
                train_exception(MSG_EXPERIMENT_MODE_EXCEPTION, train) {};
            explicit experiment_mode_exception(const std::string train_name):
                            train_exception(MSG_EXPERIMENT_MODE_EXCEPTION, train_name) {};

        };

        class synchronization_channel_number_exception: public train_exception{
        public:
            explicit synchronization_channel_number_exception(const FileTrain* train):
                train_exception(MSG_SYNCHRONIZATION_CHANNEL_NUMBER_EXCEPTION, train) {};
            explicit synchronization_channel_number_exception(const std::string& train_name):
                train_exception(MSG_SYNCHRONIZATION_CHANNEL_NUMBER_EXCEPTION, train_name) {};
        };

        class unsupported_experiment_mode_exception: public train_exception{
        public:
            explicit unsupported_experiment_mode_exception(const FileTrain* train):
                train_exception(MSG_UNSUPPORTED_EXPERIMENT_MODE_EXCEPTION,
                        train) {};
                explicit unsupported_experiment_mode_exception(const std::string& train_name):
                        train_exception(MSG_UNSUPPORTED_EXPERIMENT_MODE_EXCEPTION,
                                train_name) {};
        };

        class frame_number_mismatch: public SourceFile::source_file_exception{
        public:
            explicit frame_number_mismatch(SourceFile* file):
                SourceFile::source_file_exception(MSG_FRAME_NUMBER_MISMATCH, file) {};
            explicit frame_number_mismatch(const std::string& file_name, const std::string& train_name = ""):
                SourceFile::source_file_exception(MSG_FRAME_NUMBER_MISMATCH,
                                                  file_name, train_name) {};

        };

        class data_chunk_size_mismatch: public SourceFile::source_file_exception{
        public:
            explicit data_chunk_size_mismatch(SourceFile* file):
                    SourceFile::source_file_exception(MSG_DATA_CHUNK_SIZE_MISMATCH, file) {};
            explicit data_chunk_size_mismatch(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_DATA_CHUNK_SIZE_MISMATCH, filename, trainname) {};
        };

        class isoi_chunk_size_mismatch: public SourceFile::source_file_exception{
        public:
            explicit isoi_chunk_size_mismatch(SourceFile* file):
                    SourceFile::source_file_exception(MSG_ISOI_CHUNK_SIZE_MISMATCH, file) {};
            explicit isoi_chunk_size_mismatch(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_ISOI_CHUNK_SIZE_MISMATCH, filename, trainname) {};
        };

        class file_size_mismatch: public SourceFile::source_file_exception{
        public:
            explicit file_size_mismatch(SourceFile* file):
                    SourceFile::source_file_exception(MSG_FILE_SIZE_MISMATCH, file) {};
            explicit file_size_mismatch(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_FILE_SIZE_MISMATCH, filename, trainname) {};
        };

        class experimental_chunk_not_found: public SourceFile::source_file_exception{
        public:
            explicit experimental_chunk_not_found(SourceFile* file):
                    SourceFile::source_file_exception(MSG_EXPERIMENTAL_CHUNK_NOT_FOUND, file) {};
            explicit experimental_chunk_not_found(const std::string& filename, const std::string trainname = ""):
                        SourceFile::source_file_exception(MSG_EXPERIMENTAL_CHUNK_NOT_FOUND, filename, trainname) {};
        };

        class file_header_mismatch: public SourceFile::source_file_exception{
        public:
            explicit file_header_mismatch(SourceFile* file):
                    SourceFile::source_file_exception(MSG_FILE_HEADER_MISMATCH, file) {};
            explicit file_header_mismatch(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_FILE_HEADER_MISMATCH, filename, trainname) {};
        };

        class frame_header_mismatch: public SourceFile::source_file_exception{
        public:
            explicit frame_header_mismatch(SourceFile* file):
                    SourceFile::source_file_exception(MSG_FRAME_HEADER_MISMATCH, file) {};
            explicit frame_header_mismatch(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_FRAME_HEADER_MISMATCH, filename, trainname) {};
        };

        class map_dimensions_mismatch: public SourceFile::source_file_exception{
        public:
            explicit map_dimensions_mismatch(SourceFile* file):
                    SourceFile::source_file_exception(MSG_MAP_DIMENSIONS_MISMATCH, file) {};
            explicit map_dimensions_mismatch(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_MAP_DIMENSIONS_MISMATCH, filename, trainname) {};
        };

        class data_type_mismatch: public SourceFile::source_file_exception{
        public:
            explicit data_type_mismatch(SourceFile* file):
                    SourceFile::source_file_exception(MSG_DATA_TYPE_MISMATCH, file) {};
            explicit data_type_mismatch(const std::string& filename, const std::string& trainname = ""):
                        SourceFile::source_file_exception(MSG_DATA_TYPE_MISMATCH, filename, trainname) {};
        };

        /**
         * Creates new file train
         *
         * @param path full path to the file train
         * @param filename filename (without any extension)
         * @param traverse uses in case when filename doesn't refer to the head of the file
         * true - the file will be substituted to another file that is in the file head
         * false - TrainSourceFile::not_train_head will be thrown
         */
        FileTrain(const std::string& path, const std::string& filename, bool traverse = false):
                synchChannelMax(){
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
         *
         * @return FileTrain:;Continuous is stimulus properties (i.e., orientation, direction etc.) are changed
         * continuously
         * FileTrain::Episodic is stimulus with certain properties is presented for a certain fixed time during which
         * its properties are the same.
         */
        [[nodiscard]] ExperimentalMode getExperimentalMode() const { return experimentalMode; }

        /**
         *
         * @return map size on X, in pixels
         */
        [[nodiscard]] int getXSize() const { return xSize; }

        /**
         *
         * @return map size in Y, in pixels
         */
        [[nodiscard]] int getYSize() const { return ySize; }

        /**
         *
         * @return total number of pixels in the single frame
         */
        [[nodiscard]] int getXYSize() const { return xySize; }

        /**
         *
         * @return frame image size
         */
        [[nodiscard]] size_t getFrameImageSize() const { return frameImageSize; }

        /**
         *
         * @return total frame size
         */
        [[nodiscard]] size_t getFrameSize() const { return frameSize; }

        /**
         *
         * @return data type
         */
        [[nodiscard]] int getDataType() const { return dataType; }

        /**
         *
         * @return total number of synchronization channels
         */
        [[nodiscard]] size_t getSynchronizationChannelNumber() const{
            if (experimentalMode == Continuous){
                return synchChannelMax.size();
            } else {
                throw experiment_mode_exception(this);
            }
        };

        /**
         * Maximum value at a certain synchronization channel
         *
         * @param chan number of the synchronization channel
         * @return
         */
        [[nodiscard]] int getSynchronizationChannelMax(int chan) const{
            if (experimentalMode == Continuous){
                if (chan >= 0 && chan < synchChannelMax.size()){
                    return synchChannelMax[chan];
                } else {
                    throw synchronization_channel_number_exception(this);
                }
            } else {
                throw experiment_mode_exception(this);
            }
        }

        /**
         *
         * @return total number of frames within the record
         */
        [[nodiscard]] int getTotalFrames() const {
            return totalFrames;
        }

        /**
         * Opens all files within the file train
         */
        virtual void open();

        /**
         * Closes all files within the train
         */
        void close();

        /**
         * Moves the pointer to an appropriate frame
         *
         * @param n frame number
         * @result reference to the TrainSourceFile that contains an appropriate frame
         */
        virtual TrainSourceFile& seek(int n) = 0;

        /**
         * Reads the frame with an appropriate number
         *
         * @param n number of the frame
         * @return poitner to the frame read. The object will be constructed during the routine execution.
         * The object will not be deleted. This is your responsibility to delete the object
         * In case of throwing exception, the object will be deleted
         */
        Frame* readFrame(int n);
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAIN_H
