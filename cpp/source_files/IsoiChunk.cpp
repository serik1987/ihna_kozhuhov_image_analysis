//
// Created by serik1987 on 16.12.2019.
//

#include "IsoiChunk.h"

namespace iman{

    void IsoiChunk::readFromFile(SourceFile &file) {
        ChunkHeader chunkHeader = ChunkHeader::InvalidChunk();

        file.getFileStream().seekg(sizeof(ChunkHeader::DATA_CHUNK), std::ios_base::beg);
        file.getFileStream().read(Tag, 4);
        if (file.getFileStream().fail()){
            throw SourceFile::file_read_exception(&file);
        }
        while (true){
            chunkHeader = file.readChunkHeader();
            try {
                if (!chunkHeader.isKnown()) {
                    throw SourceFile::unsupported_chunk_exception(&file, chunkHeader.getChunkIdRaw());
                }
            } catch (ChunkHeader::chunk_size_mismatch_exception& e){
                throw SourceFile::chunk_size_mismatch_exception(&file, chunkHeader.getChunkIdRaw());
            }
            if (chunkHeader == ChunkHeader::DATA_CHUNK_CODE){
                break;
            }
            if (chunkHeader == ChunkHeader::SOFT_CHUNK_CODE){
                file.getFileStream().seekg(chunkHeader.getChunkSize(), std::ios_base::cur);
                if (file.getFileStream().fail()){
                    throw SourceFile::file_read_exception(&file);
                }
                continue;
            }
            auto* chunk = chunkHeader.createChunk();
            if (chunk == nullptr){
                throw std::runtime_error("Developmental error: ChunkHeader::createChunk doesn't create the chunk " +
                chunkHeader.getChunkId());
            }
            header_chunks.push_back(chunk);
            chunk_map[(uint32_t)chunkHeader] = chunk;
            chunk->readFromFile(file);
        }
        softChunk = &file.getSoftChunk();
        chunk_map[ChunkHeader::SOFT_CHUNK_CODE] = (Chunk*)softChunk;
        dataChunk = dynamic_cast<DataChunk*>(chunkHeader.createChunk());
        chunk_map[ChunkHeader::DATA_CHUNK_CODE] = (Chunk*)dataChunk;
    }

    IsoiChunk::~IsoiChunk() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE ISOI CHUNK\n";
#endif
        for (auto pchunk: header_chunks){
            delete pchunk;
        }
        delete dataChunk;
    }

    std::ostream &operator<<(std::ostream &out, const IsoiChunk &isoi) {
        using std::endl;

        out << "===== ISOI =====\n";
        out << "Chunk size: " << isoi.getSize() << endl;
        out << "Chunk tag: " << isoi.getTag() << endl;
        out << "All chunks containing in the ISOI chunk are given below:\n";
        for (auto it = isoi.cbegin(); it != isoi.cend(); ++it){
            out << *it << endl;
        }

        return out;
    }

    Chunk *IsoiChunk::getChunkById(uint32_t chunkId) {
        Chunk* chunk_found = nullptr;

        if (chunkId == ChunkHeader::ISOI_CHUNK_CODE){
            chunk_found = this;
        } else {
            auto it = chunk_map.find(chunkId);
            if (it != chunk_map.end()) {
                chunk_found = it->second;
            }
        }

        return chunk_found;
    }
}