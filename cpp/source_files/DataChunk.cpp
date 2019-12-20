//
// Created by serik1987 on 16.12.2019.
//

#include "DataChunk.h"

namespace GLOBAL_NAMESPACE{

    std::ostream &operator<<(std::ostream &out, const DataChunk &chunk) {
        out << "===== DATA =====\n";
        out << "Chunk size, bytes: " << chunk.getSize();

        return out;
    }
}