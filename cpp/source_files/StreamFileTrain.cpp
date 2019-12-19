//
// Created by serik1987 on 18.12.2019.
//

#include "StreamFileTrain.h"
#include "IsoiChunk.h"

namespace ihna::kozhukhov::image_analysis{

    uint32_t StreamFileTrain::getDesiredIsoiChunkSize(TrainSourceFile &file) {
        uint32_t desired_size;
        desired_size = file.getIsoiChunk().getChunkById(ChunkHeader::DATA_CHUNK_CODE)->getSize() +
                file.getFileHeaderSize() - sizeof(ChunkHeader::DATA_CHUNK);

        return desired_size;
    }

    uint32_t StreamFileTrain::getFileSizeChecksum(TrainSourceFile &file) {
        file.setFileSize(file_sizes[file_idx++]);

        return file.fileSizeCheck();
    }
}