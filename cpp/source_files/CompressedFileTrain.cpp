//
// Created by serik1987 on 18.12.2019.
//

#include "CompressedFileTrain.h"
#include "IsoiChunk.h"

namespace GLOBAL_NAMESPACE{

    uint32_t CompressedFileTrain::getDesiredIsoiChunkSize(TrainSourceFile& file) {
        uint32_t desired_size;
        Chunk* compChunk = file.getIsoiChunk().getChunkById(ChunkHeader::COMP_CHUNK_CODE);
        if (compChunk == nullptr){
            throw comp_chunk_not_exist_exception(&file);
        }
        desired_size = file.getIsoiChunk().getChunkById(ChunkHeader::DATA_CHUNK_CODE)->getSize() +
                file.getFileHeaderSize() - compChunk->getSize() - 2 * sizeof(ChunkHeader::DATA_CHUNK);

        return desired_size;
    }

    TrainSourceFile& CompressedFileTrain::seek(int n) {
        if (n != 0){
            throw compressed_frame_read_exception(this);
        }

        auto* pfile = *begin();
        pfile->getFileStream().seekg(getFileHeaderSize(), std::ios_base::beg);
        return *pfile;
    }
}