//
// Created by serik1987 on 15.12.2019.
//

#include "Chunk.h"

#include "SoftChunk.h"
#include "IsoiChunk.h"
#include "CompChunk.h"
#include "HardChunk.h"
#include "CostChunk.h"
#include "EpstChunk.h"

namespace ihna::kozhukhov::image_analysis{

    void Chunk::readFromFile(SourceFile &file) {
        file.getFileStream().read(body, getSize());
        if (file.getFileStream().fail()){
            throw SourceFile::file_read_exception(&file);
        }
    }

    std::ostream &operator<<(std::ostream &out, const Chunk &chunk) {
        const auto* softChunk = dynamic_cast<const SoftChunk*>(&chunk);
        const auto* isoiChunk = dynamic_cast<const IsoiChunk*>(&chunk);
        const auto* compChunk = dynamic_cast<const CompChunk*>(&chunk);
        const auto* hardChunk = dynamic_cast<const HardChunk*>(&chunk);
        const auto* costChunk = dynamic_cast<const CostChunk*>(&chunk);
        const auto* epstChunk = dynamic_cast<const EpstChunk*>(&chunk);
        const auto* dataChunk = dynamic_cast<const DataChunk*>(&chunk);

        if (softChunk != nullptr) {
            out << *softChunk;
        } else if (isoiChunk != nullptr) {
            out << *isoiChunk;
        } else if (compChunk != nullptr) {
            out << *compChunk;
        } else if (hardChunk != nullptr) {
            out << *hardChunk;
        } else if (costChunk != nullptr) {
            out << *costChunk;
        } else if (epstChunk != nullptr) {
            out << *epstChunk;
        } else if (dataChunk != nullptr) {
            out << *dataChunk;
        } else {
            out << "===== The chunk is unsupported or its write is not implemented =====\n";
            out << "Chunk name: " << chunk.getName() << std::endl;
            out << "Chunk size: " << chunk.getSize();
        }

        return out;
    }
}