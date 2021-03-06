//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H


#include <fstream>
#include "../exceptions.h"
#include "ChunkHeader.h"


namespace GLOBAL_NAMESPACE {

    class ChunkHeader;
    class Chunk;
    class SoftChunk;
    class IsoiChunk;

    /**
     * This is the base class for the IMAN source file
     */
    class SourceFile {
    public:
        enum ChunkPositionPointer {PositionStartHeader, PositionFinishHeader, PositionFinishChunk};
        enum FileType {AnalysisFile, CompressedFile, GreenFile, StreamFile, UnknownFile};

    private:
        std::ifstream fileStream;
        bool fileStatus;
        std::string filePath;
        std::string fileName;
        std::string fullName;
        std::string trainName;

        static const int CHUNK_ID_SIZE;

        bool loaded = false;

        uint32_t frameHeaderSize = -1;
        std::ios::pos_type fileHeaderSize = -1;
        FileType fileType = FileType::UnknownFile;
        uint32_t file_size = 0;

    protected:
        SoftChunk* softChunk = nullptr;
        IsoiChunk* isoiChunk = nullptr;
        void setName(const std::string& newName){
            fileName = newName;
            fullName = filePath + fileName;
        }

    public:
        class source_file_exception: public io_exception{
        private:
            std::string train_name;

        public:
            source_file_exception(const std::string& message, SourceFile* parent):
                io_exception(message, parent->getFullname()), train_name(parent->trainName) {};
            source_file_exception(const std::string& message, const std::string& filename,
                    const std::string train_name = ""):
                io_exception(message, filename), train_name(train_name) {};

            [[nodiscard]] const std::string& getTrainName() const { return train_name; }
        };

        class file_open_exception: public source_file_exception{
        public:
            explicit file_open_exception(SourceFile* parent):
                source_file_exception(MSG_FILE_OPEN_EXCEPTION, parent) {};
            explicit file_open_exception(const std::string& filename, const std::string& trainname = ""):
                    source_file_exception(MSG_FILE_OPEN_EXCEPTION, filename, trainname) {};
        };

        class file_read_exception: public source_file_exception{
        public:
            explicit file_read_exception(SourceFile* parent):
                source_file_exception(MSG_FILE_READ_EXCEPTION, parent) {};
            explicit file_read_exception(const std::string& filename, const std::string trainname = ""):
                    source_file_exception(MSG_FILE_READ_EXCEPTION, filename, trainname) {};
        };

        class file_write_exception: public source_file_exception{
        public:
            explicit file_write_exception(const std::string& filename):
                source_file_exception("Error in writing to the output file", filename) {};
        };

        class unsupported_chunk_exception: public source_file_exception{
        private:
            const char* chunk_name = NULL;

        public:
            unsupported_chunk_exception(SourceFile* parent, const char* id):
                source_file_exception(MSG_UNSUPPORTED_CHUNK_EXCEPTION, parent),
                chunk_name(id){};
            unsupported_chunk_exception(const char* id, const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_UNSUPPORTED_CHUNK_EXCEPTION,
                filename, trainname), chunk_name(id) {};

            [[nodiscard]] const char* getChunkName() const { return chunk_name; }
        };

        class chunk_size_mismatch_exception: public source_file_exception{
        private:
            const char* chunk_name;

        public:
            chunk_size_mismatch_exception(SourceFile* parent, const char* id):
                source_file_exception(MSG_CHUNK_SIZE_MISMATCH_EXCEPTION, parent), chunk_name(id) {};
            chunk_size_mismatch_exception(const char* id, const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_CHUNK_SIZE_MISMATCH_EXCEPTION, filename, trainname),
                chunk_name(id) {};

            [[nodiscard]] const char* getChunkName() const { return chunk_name; }
        };

        class chunk_not_found_exception: public source_file_exception{
        private:
            const char* chunk_name;

        public:
            chunk_not_found_exception(SourceFile* parent, const std::string& name):
                source_file_exception(MSG_CHUNK_NOT_FOUND_EXCEPTION, parent),
                chunk_name(name.c_str()) {};
            chunk_not_found_exception(const std::string& name, const std::string& filename,
                    const std::string& trainname = ""):
                source_file_exception(MSG_CHUNK_NOT_FOUND_EXCEPTION, filename, trainname),
                chunk_name(name.c_str()) {};

            [[nodiscard]] const char* getChunkName() const { return chunk_name; }
        };

        class file_not_opened: public source_file_exception{
        public:
            file_not_opened(SourceFile* parent, const std::string& operation):
                source_file_exception(MSG_FILE_NOT_OPENED_EXCEPTION, parent) {};
            file_not_opened(const std::string& operation, const std::string& filename,
                    const std::string& trainname = ""):
                    source_file_exception(MSG_FILE_NOT_OPENED_EXCEPTION,
                            filename, trainname) {};
        };

        class file_not_isoi_exception: public source_file_exception{
        public:
            file_not_isoi_exception(SourceFile* parent):
                source_file_exception(MSG_FILE_NOT_ISOI_EXCEPTION, parent) {};
            file_not_isoi_exception(const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_FILE_NOT_ISOI_EXCEPTION, filename, trainname) {};
        };

        class file_not_loaded_exception: public source_file_exception{
        public:
            file_not_loaded_exception(const std::string& methodName, SourceFile* parent):
                source_file_exception(MSG_FILE_NOT_LOADED_EXCEPTION, parent) {};
            file_not_loaded_exception(const std::string& methodName, const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_FILE_NOT_LOADED_EXCEPTION, filename, trainname) {};
        };

        class data_chunk_not_found_exception: public source_file_exception{
        public:
            explicit data_chunk_not_found_exception(SourceFile* parent):
                source_file_exception(MSG_DATA_CHUNK_NOT_FOUND_EXCEPTION, parent) {};
            explicit data_chunk_not_found_exception(const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_DATA_CHUNK_NOT_FOUND_EXCEPTION, filename, trainname) {};
        };

        /**
         * Creates an instance for the source file
         *
         * @param path path to the source file, doesn't contain the filename
         * @param name name of the source file
         * @param trainName if available, point out the train name. This option influences only on the information
         * containing in error messages, the package functionality is absolutely unchangeable by this
         */
        SourceFile(const std::string& path, const std::string& name, const std::string& train_name = "");

        /**
         * Destructs the source file
         */
        virtual ~SourceFile();

        SourceFile(const SourceFile& other) = delete;
        SourceFile& operator=(SourceFile&) = delete;

        /**
         *
         * @return path to the file
         */
        [[nodiscard]] const std::string& getFilePath() const { return filePath; }

        /**
         *
         * @return name of the file
         */
        [[nodiscard]] const std::string& getFileName() const { return fileName; }

        /**
         *
         * @return full name of the file (including the file path)
         */
        [[nodiscard]] const std::string& getFullname() const { return fullName; }

        /**
         *
         * @return true is the file is opened
         */
        [[nodiscard]] bool isOpened() const { return fileStatus; }

        /**
         *
         * @return the file stream
         */
        [[nodiscard]] std::ifstream& getFileStream() { return fileStream; }

        /**
         * Opens the IMAN source file
         */
        void open();

        /**
         * Closes the IMAN source file
         */
        void close();

        /**
         * Reads the chunk header and move the file to the initial position
         *
         * See Chunk help for details
         *
         * @param return_position
         * @return reference to the chunk header
         */
        ChunkHeader readChunkHeader(bool return_position = false);

        /**
         * Looks through the whole file and find chunk with an appropriate name. Moves the file pointer to this chunk
         * When the chunk is not found, the function returns an invalid chunk.
         * The file shall be positioned at the beginning of any chunk header. Otherwise, no chunks can be found.
         * All chunks before the current position will be ignored
         *
         * See Chunk help for details
         *
         * @param name chunk name as string. The name shall refere to the name of the valid chunk. If the chunk with
         * a certain name is not suported by the ChunkHeader, the method will not work
         * @param pointer defines the place where the pointer shall be located:
         * PositionStartHeader - place the pointer at the beginning of the chunk
         * PositionFinishHeader - place the pointer between the chunk header and the chunk body
         * PositionFinishChunk - place the pointer at the end of the chunk
         * @param originalReturnOnFail when true, returns the pointer to the original position due to logical error
         * Logical errors are assumed to be any exceptions except file_read_exception (when fail() is true)
         * @return ChunkHeader structure. if the routine is unable to find the chuk the function returns
         * InvalidChunkHeader.
         * @throw file_read_exception is file reading or file seeking is failed
         *        ChunkHeader::unsupported_chunk_exception if name contains name of an unsupported chunk
         *        unsupported_chunk_exception if unsupported chunk was found in the file
         *
         */
        ChunkHeader findChunkHeader(const std::string& name, ChunkPositionPointer pointer = PositionFinishHeader,
                bool originalReturnOnFail = false);

        /**
         * Looks through the whole file trying to find and appropriate chunk. Once the chunk is found it will be loaded
         * into the file.
         * The file shall be positioned at the beginning of any chunk header. Otherwise, no chunks can be found.
         * All chunks before the current position will be ignored.
         * At the end of the successful finish of the program the file pointer will be positioned at the end of the
         * chunk
         *
         * @param name chunk name as string. The name shall refere to the name of the valid chunk. If the chunk with
         * a certain name is not suported by the ChunkHeader, the method will not work
         * @param originalReturnOnFail if true, the pointer will be positioned at finish of the chunk in case of failure
         * to find the chunk
         * @param chunkIsOptional if true, the method will return nullptr in failure to find the chunk. If false,
         * the methon will throw the chunk_not_found exception
         * @return pointer to the chunk created or nullptr in some cases mentioned above (running the method with
         * default arguments will never get nullptr). The chunk will be created as dynamic object and this will not
         * be delete during finish of the method execution. This is your responsibility to delete an object
         */
        Chunk* findChunk(const std::string& name, bool originalReturnOnFail = false, bool chunkIsOptional = false);

        /**
         *
         * @return true if the file has been loaded. The file is considered to be loaded if its general information
         * is loaded
         */
        bool isLoaded() const { return loaded; }

        /**
         * Loads the general info from the file (generally, loads information from SOFT and DATA chunks).
         */
        virtual void loadFileInfo();

        /**
         *
         * @return description of the current file type. The current file reflects the base class this method belongs
         * to, not value within the SOFT chunk.
         *
         * EXAMPLE:
         * SourceFile f("", "T_1BF.0A01z");
         * sourceFile.open();
         * sourceFile.loadFileInfo();
         * Will get the file type called "based file". No operations except reading the file header are possible.
         * This is because you used 'SourceFile'. Use another class derived from the 'SourceFile' is you want
         * more functions.
         *
         * Yes, don't try to call: train.addFile(*(StreamFile*)&f);
         * The program will be simply crashed!
         *
         * EXAMPLE 2:
         * StreamFile f("", "T_1BF.0A01z");
         * f.open();
         * has a type "stream file", even though the T_1BF.0A01z's type is defined as "compressed file. The following
         * code will through an exception:
         * f.laodFileInfo();
         *
         * EXAMPLE 3:
         * CompressedFile f("", "T_1BF.0A01z");
         * f.open();
         * f.loadFileInfo();
         * will have a type "compressed file", the code above will not generate an exception
         */
        virtual std::string getFileTypeDescription() const { return "base file [available file view only]"; }

        /**
         * Prints general information about the file into arbitrary stream
         *
         * @param out the stream that accepts information about the file
         * @param file the file itself
         * @return reference to out
         */
        friend std::ostream& operator<<(std::ostream& out, const SourceFile& file);

        /**
         *
         * @return reference to the SOFT chunk
         */
        SoftChunk& getSoftChunk();

        /**
         *
         * @return ISOI chunk
         */
        IsoiChunk& getIsoiChunk();

        /**
         *
         * @return size of the frame header. When the file is not loaded by means of loadFileInfo the function
         * returns -1
         */
        [[nodiscard]] uint32_t getFrameHeaderSize() const { return frameHeaderSize; }

        /**
         *
         * @return size of the file header. When the file is not loaded by means of loadFileInfo the function
         * returns -1
         */
        [[nodiscard]] std::ios::pos_type getFileHeaderSize() const { return fileHeaderSize; }

        /**
         *
         * @return the file type or UnknownFileType is the file is not loaded by means of loadFileinfo function
         */
        [[nodiscard]] FileType getFileType() const { return fileType; }

        /**
         * Sets the actual file size.
         * The function shall be called before the train open by some platform-dependent of Python code in order
         * to set the file size. Failure to do this will throw an exception
         *
         * @param size
         */
        void setFileSize(uint32_t size){
            file_size = size;
        }

        /**
         * When the function return is different from zero the program execution will be failed.
         *
         * @return
         */
        uint32_t fileSizeCheck();
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H
