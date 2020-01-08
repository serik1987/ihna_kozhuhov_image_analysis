//
// Created by serik1987 on 16.12.2019.
//

#include "CompChunk.h"

namespace GLOBAL_NAMESPACE{

    std::ostream &operator<<(std::ostream &out, const CompChunk &chunk) {
        using std::endl;

        out << "===== COMP =====\n";
        out << "Size of one single extra pixel, bytes: " << chunk.getCompressedRecordSize() << endl;
        out << "Size of a single compressed frame, bytes: " << chunk.getCompressedFrameSize() << endl;
        out << "Number of compressed frames: " << chunk.getCompressedFrameNumber();

        return out;
    }

    CompChunk::CompChunk(uint32_t compressed_record_size, uint32_t compressed_frame_size,
                         uint32_t compressed_frame_number): Chunk("COMP", sizeof(COMP_CHUNK)) {
        info.CompressedRecordSize = compressed_record_size;
        info.CompressedFrameSize = compressed_frame_size;
        info.CompressedFrameNumber = compressed_frame_number;
        strncpy(info.Tag, "0000", 4);
        info.Free[0] = 0;   info.Free[1] = 0;   info.Free[2] = 0;
        body = (char*)&info;
    }
}