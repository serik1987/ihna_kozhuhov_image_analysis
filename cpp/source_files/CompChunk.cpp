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
}