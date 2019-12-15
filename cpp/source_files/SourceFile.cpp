//
// Created by serik1987 on 15.12.2019.
//

#include <iostream>
#include <algorithm>
#include "SourceFile.h"

namespace iman{


    SourceFile::SourceFile(const std::string &path, const std::string &name) {
        filePath = path;
        fileName = name;
        fullName = path + name;
        fileStatus = false;
    }

    SourceFile::~SourceFile() {
        if (isOpened()){
            close();
        }
    }

    void SourceFile::open() {
        fileStream.open(fullName, std::ios_base::binary);
        if (fileStream.fail()){
            std::string originalName = fullName;
            std::transform(fileName.begin(), fileName.end(), fileName.begin(), [](unsigned char c){
                return std::tolower(c);
            });
            setName(fileName);
            fileStream.open(fullName, std::ios_base::binary);
            if (fileStream.fail()){
                std::transform(fileName.begin(), fileName.end(), fileName.begin(), [](unsigned char c){
                    return std::toupper(c);
                });
                setName(fileName);
                fileStream.open(fullName, std::ios_base::binary);
                if (fileStream.fail()){
                    setName(originalName);
                    throw file_open_exception(this);
                }
            }
        }
        fileStatus = true;
    }

    void SourceFile::close() {
        fileStream.close();
        fileStatus = false;
    }

    ChunkHeader SourceFile::readChunkHeader(bool return_position) {
        ChunkHeader::DATA_CHUNK body = {"", 0};
        constexpr size_t body_size = sizeof(body);

        fileStream.read((char*)&body, body_size);
        if (return_position) {
            fileStream.seekg(-body_size, std::ios_base::cur);
        }
        if (fileStream.fail()){
            throw file_read_exception(this);
        }

        return ChunkHeader(body);
    }

    ChunkHeader SourceFile::findChunkHeader(const std::string &name, SourceFile::ChunkPositionPointer pointer,
                                            bool originalReturnOnFail) {
        auto header = ChunkHeader::InvalidChunk();
        auto original_pos = fileStream.tellg();

        if (!ChunkHeader::isKnown(name)){
            throw ChunkHeader::unsupported_chunk_exception(name.c_str());
        }
#ifdef DEBUG_CHUNK_LIST_DISPLAY
        std::cout << "Name\tSize\n";
        std::cout << "==============================\n";
#endif
        while (!fileStream.eof()){
            auto local_header = readChunkHeader(false);
#ifdef DEBUG_CHUNK_LIST_DISPLAY
            std::cout << local_header.getChunkId() << "\t" << local_header.getChunkSize() << "\n";
#endif
            if (!local_header.isKnown()){
                throw unsupported_chunk_exception(this, local_header.getChunkIdRaw());
            }
            if (local_header == ChunkHeader::ISOI_CHUNK_CODE){
                if (name == "ISOI"){
                    header = local_header;
                    break;
                } else {
                    fileStream.seekg(ChunkHeader::CHUNK_TAG_SIZE, std::ios_base::cur);
                    if (fileStream.fail()){
                        throw file_read_exception(this);
                    }
                }
            } else {
                if (local_header == name){
                    header = local_header;
                    break;
                } else {
                    fileStream.seekg(local_header.getChunkSize(), std::ios_base::cur);
                    if (fileStream.fail()){
                        throw file_read_exception(this);
                    }
                }
            }
            if (local_header == ChunkHeader::DATA_CHUNK_CODE){
                break;
            }
        }
#ifdef DEBUG_CHUNK_LIST_DISPLAY
        std::cout << "==============================\n";
#endif

        if (header.isInvalid()){
#ifdef DEBUG_CHUNK_LIST_DISPLAY
            std::cout << "HEADER NOT FOUND\n";
#endif
            if (originalReturnOnFail){
                fileStream.seekg(original_pos);
                if (fileStream.fail()){
                    throw file_read_exception(this);
                }
            }
        }

        if (header.isValid()){
#ifdef DEBUG_CHUNK_LIST_DISPLAY
            std::cout << "HEADER FOUND\n";
#endif
            if (pointer == PositionStartHeader){
                fileStream.seekg(-sizeof(ChunkHeader::DATA_CHUNK), std::ios_base::cur);
            }
            if (pointer == PositionFinishChunk){
                fileStream.seekg(header.getChunkSize(), std::ios_base::cur);
            }
            if (fileStream.fail()){
                throw file_read_exception(this);
            }
        }

        return header;
    }
}
