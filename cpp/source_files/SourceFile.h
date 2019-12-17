//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H


#include <fstream>
#include "../core.h"
#include "ChunkHeader.h"

namespace iman {

    class ChunkHeader;
    class Chunk;

    /**
     * This is the base class for the IMAN source file
     */
    class SourceFile {
    private:
        std::ifstream fileStream;
        bool fileStatus;
        std::string filePath;
        std::string fileName;
        std::string fullName;

        void setName(const std::string& newName){
            fileName = newName;
            fullName = filePath + fileName;
        }

        static const int CHUNK_ID_SIZE;

    public:
        enum ChunkPositionPointer {PositionStartHeader, PositionFinishHeader, PositionFinishChunk};

        class source_file_exception: public io_exception{
        public:
            source_file_exception(const std::string& message, SourceFile* parent):
                io_exception(message, parent->getFullname()) {};
        };

        class file_open_exception: public source_file_exception{
        public:
            explicit file_open_exception(SourceFile* parent):
                source_file_exception("Error in opening the file", parent) {};
        };

        class file_read_exception: public source_file_exception{
        public:
            explicit file_read_exception(SourceFile* parent):
                source_file_exception("Error in reading the file", parent) {};
        };

        class unsupported_chunk_exception: public source_file_exception{
        public:
            unsupported_chunk_exception(SourceFile* parent, const char* id):
                source_file_exception("Chunk '" + std::string(id, CHUNK_ID_SIZE) +
                "' is presented in the file but not supported by the current version of image-analysis", parent) {};
        };

        class chunk_size_mismatch_exception: public source_file_exception{
        public:
            chunk_size_mismatch_exception(SourceFile* parent, const char* id):
                source_file_exception("Chunk '" + std::string(id, CHUNK_ID_SIZE) +
                "' has an actual size that is much different than the desired size", parent) {};
        };

        class chunk_not_found_exception: public source_file_exception{
        public:
            chunk_not_found_exception(SourceFile* parent, const std::string& name):
                source_file_exception("Chunk '" + name + "' was not found in the source file", parent) {};
        };

        /**
         * Creates an instance for the source file
         *
         * @param path path to the source file, doesn't contain the filename
         * @param name name of the source file
         */
        SourceFile(const std::string& path, const std::string& name);

        /**
         * Destructs the source file
         */
        ~SourceFile();

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
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H
