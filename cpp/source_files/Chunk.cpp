//
// Created by serik1987 on 15.12.2019.
//

#include "Chunk.h"

#include "SoftChunk.h"
#include "IsoiChunk.h"

namespace iman{

    void Chunk::readFromFile(SourceFile &file) {
        file.getFileStream().read(body, getSize());
        if (file.getFileStream().fail()){
            throw SourceFile::file_read_exception(&file);
        }
    }

    std::ostream &operator<<(std::ostream &out, const Chunk &chunk) {
        const auto* softChunk = dynamic_cast<const SoftChunk*>(&chunk);
        const auto* isoiChunk = dynamic_cast<const IsoiChunk*>(&chunk);

        if (softChunk != nullptr) {
            out << *softChunk;
        } else if (isoiChunk != nullptr){
            out << *isoiChunk;
        } else {
            out << "===== The chunk is unsupported or its write is not implemented =====\n";
            out << "Chunk name: " << chunk.getName() << std::endl;
            out << "Chunk size: " << chunk.getSize();
        }

        return out;
    }
}