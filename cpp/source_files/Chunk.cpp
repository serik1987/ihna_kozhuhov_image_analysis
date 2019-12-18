//
// Created by serik1987 on 15.12.2019.
//

#include "Chunk.h"

#include "SoftChunk.h"

namespace iman{

    void Chunk::readFromFile(SourceFile &file) {
        file.getFileStream().read(body, getSize());
        if (file.getFileStream().fail()){
            throw SourceFile::file_read_exception(&file);
        }
    }

    std::ostream &operator<<(std::ostream &out, const Chunk &chunk) {
        auto* softChunk = (SoftChunk*)&chunk;

        if (softChunk != nullptr){
            out << *softChunk;
        } else {
            out << "===== The chunk is unsupported or its write is not implemented =====";
            out << "Chunk name: " << chunk.getName() << std::endl;
            out << "Chunk size: " << chunk.getSize() << std::endl;
        }

        return out;
    }
}