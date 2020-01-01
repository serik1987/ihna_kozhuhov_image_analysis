//
// Created by serik1987 on 18.12.2019.
//

#include "StreamFileTrain.h"
#include "IsoiChunk.h"
#include "Frame.h"

namespace GLOBAL_NAMESPACE{

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

    TrainSourceFile& StreamFileTrain::seek(int n) {
        if (n < 0 || n >= getTotalFrames()){
            throw Frame::frame_is_out_of_range(this, n);
        }
        if (!isOpened()){
            throw SourceFile::file_not_opened("seek", getFilePath() + getFilename(), getFilePath() + getFilename());
        }
        TrainSourceFile* pfile = nullptr;
        int ini_frame = -1;
        int fin_frame = -1;
        for (auto* file: *this){
            if (pfile == nullptr){
                pfile = file;
                ini_frame = pfile->offsetFrame;
            } else {
                fin_frame = file->offsetFrame - 1;
                if (n >= ini_frame && n <= fin_frame){
                    break;
                }
                pfile = file;
                ini_frame = pfile->offsetFrame;
            }
        }
        if (pfile == nullptr){
            throw std::runtime_error("pfile is nullptr");
        }
        int relative_frame = n - ini_frame;
        unsigned long position = (unsigned long)relative_frame * getFrameSize() + getFileHeaderSize();
        pfile->getFileStream().seekg(position, std::ios_base::beg);
        if (pfile->getFileStream().fail()){
            throw SourceFile::file_read_exception(pfile);
        }

        return *pfile;
    }
}