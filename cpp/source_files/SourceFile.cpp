//
// Created by serik1987 on 15.12.2019.
//

#include <iostream>
#include <algorithm>
#include "SourceFile.h"
#include "SoftChunk.h"
#include "IsoiChunk.h"

namespace GLOBAL_NAMESPACE{

    const int SourceFile::CHUNK_ID_SIZE = ChunkHeader::CHUNK_ID_SIZE;


    SourceFile::SourceFile(const std::string &path, const std::string &name, const std::string& train_name) {
        filePath = path;
        fileName = name;
        fullName = path + name;
        trainName = train_name;
        fileStatus = false;
    }

    SourceFile::~SourceFile() {
        delete softChunk;
        delete isoiChunk;
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
#ifdef DEBUG_DELETE_CHECK
        std::cout << "CLOSING SOURCE FILE\n";
#endif
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
            auto local_header = readChunkHeader(false); // @throws file_read_exception
#ifdef DEBUG_CHUNK_LIST_DISPLAY
            std::cout << local_header.getChunkId() << "\t" << local_header.getChunkSize() << "\n";
#endif
            try {
                if (!local_header.isKnown()) {
                    throw unsupported_chunk_exception(this, local_header.getChunkIdRaw());
                }
            } catch (ChunkHeader::chunk_size_mismatch_exception& e){
                throw chunk_size_mismatch_exception(this, local_header.getChunkIdRaw());
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

    Chunk *SourceFile::findChunk(const std::string &name, bool originalReturnOnFail, bool chunkIsOptional) {
        Chunk* chunk = nullptr;
        auto chunk_header = findChunkHeader(name, PositionFinishHeader, originalReturnOnFail);
        chunk = chunk_header.createChunk();
        if (chunk == nullptr && !chunkIsOptional){
            throw chunk_not_found_exception(this, name);
        }
        if (chunk != nullptr) {
            chunk->readFromFile(*this);
        }

        return chunk;
    }

    std::ostream &operator<<(std::ostream &out, const SourceFile &file) {
        using std::endl;

        out << "===== " << file.getFileName() << " =====\n";
        out << "File path: " << file.getFilePath() << endl;
        out << "File status: ";
        if (file.isOpened()){
            out << "opened";
        } else {
            out << "closed";
        }
        if (file.isLoaded()){
            out << ", loaded\n";
        } else {
            out << "\n";
        }
        out << "Desired file type: " << file.getFileTypeDescription();
        if (file.isLoaded()){
            out << endl;
            out << "Frame header size: " << file.getFrameHeaderSize() << endl;
            out << "File header size: " << file.getFileHeaderSize() << endl;
            out << "Actual file type: ";
            switch (file.getFileType()){
                case SourceFile::AnalysisFile:
                    out << "Analysis file";
                    break;
                case SourceFile::CompressedFile:
                    out << "Compressed file";
                    break;
                case SourceFile::GreenFile:
                    out << "Green file";
                    break;
                case SourceFile::StreamFile:
                    out << "Stream file";
                    break;
                case SourceFile::UnknownFile:
                    out << "Unknown or unsupported file";
                    break;
                default:
                    out << "[TO-DO: UPDATE operator<<(std::ostream&, const iman::SourceFile&) IN SourceFile.cpp]";
                    break;
            }
        }

        return out;
    }

    SoftChunk& SourceFile::getSoftChunk() {
        if (softChunk == nullptr){
            throw file_not_loaded_exception("getSoftChunk", this);
        }
        return *softChunk;
    }

    void SourceFile::loadFileInfo() {
        if (!isOpened()){
            throw file_not_opened(this, "loadFileInfo");
        }
        fileStream.seekg(0, std::ios_base::beg);
        if (fileStream.fail()){
            throw file_read_exception(this);
        }
        auto header = readChunkHeader(true);
        if (header != ChunkHeader::ISOI_CHUNK_CODE){
            throw file_not_isoi_exception(this);
        }
        isoiChunk = (IsoiChunk*)header.createChunk();
        softChunk = (SoftChunk*)findChunk("SOFT");
        frameHeaderSize = softChunk->getFrameHeaderSize();
        header = findChunkHeader("DATA");
        if (header != ChunkHeader::DATA_CHUNK_CODE){
            throw data_chunk_not_found_exception(this);
        }
        fileHeaderSize = fileStream.tellg();
        fileType = softChunk->getFileType();
        loaded = true;
    }

    IsoiChunk &SourceFile::getIsoiChunk() {
        if (isoiChunk == nullptr){
            throw file_not_loaded_exception("getIsoiChunk", this);
        }
        return *isoiChunk;
    }

    uint32_t SourceFile::fileSizeCheck() {
        return getIsoiChunk().getSize() + sizeof(ChunkHeader::DATA_CHUNK) - file_size;
    }
}
