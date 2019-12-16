//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H


#include <fstream>
#include "../core.h"
#include "ChunkHeader.h"

namespace iman {

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
                source_file_exception("Chunk '" + std::string(id, ChunkHeader::CHUNK_ID_SIZE) +
                "' is presented in the file but not supported by the current version of image-analysis", parent) {};
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
         * Also, the chunk shall be treated as a valid chunk.
         * When the chunk is not found, the function returns an invalid chunk.
         * The file shall be positioned at the beginning of any chunk header. Otherwise, no chunks can be found.
         * All chunks before the current position will be ignored
         *
         * See Chunk help for details
         *
         * @param name chunk name as string
         * @param pointer defines the place where the pointer shall be located:
         * PositionStartHeader - place the pointer at the beginning of the chunk
         * PositionFinishHeader - place the pointer between the chunk header and the chunk body
         * PositionFinishChunk - place the pointer at the end of the chunk
         * @param originalReturnOnFail when true, returns the pointer to the original position after reading has been
         * finished
         * @return ChunkHeader structure. if the routine is unable to find the chuk the function returns
         * InvalidChunkHeader. For such a header isInvalid() always returns true while for any other chunk header
         * isInvalid() always returns false
         * @throw file_read_exception is file reading or file seeking is failed
         *        ChunkHeader::unsupported_chunk_exception if name contains name of an unsupported chunk
         *        unsupported_chunk_exception if unsupported chunk was found in the file
         *
         */
        ChunkHeader findChunkHeader(const std::string& name, ChunkPositionPointer pointer = PositionFinishHeader,
                bool originalReturnOnFail = false);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOURCEFILE_H
