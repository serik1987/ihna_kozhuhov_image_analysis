//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_ISOICHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_ISOICHUNK_H

#include "Chunk.h"

namespace iman {

    /**
     * This is the Main Container Chunk that contains all other chunks
     * presented in the file.
     *
     * The size of the ISOI chunk is the same as the file size. However, when the file
     * is compressed, actual size of ISOI chunk is not the same as value defined in the
     * header of this chunk, such value reflects the file size in the uncompressed state
     */
    class IsoiChunk: public Chunk {
    private:
        static constexpr int ISOI_CHUNK_TAG_SIZE = 4;

        char Tag[ISOI_CHUNK_TAG_SIZE] = "\x00\x00\x00";
    public:
        explicit IsoiChunk(uint32_t size): Chunk("ISOI", size){
            body = nullptr;
        };

        /**
         *
         * @return the ISOI chunk tag transformed into char
         */
        [[nodiscard]] std::string getTag() const {
            return std::string(Tag, ISOI_CHUNK_TAG_SIZE);
        };
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_ISOICHUNK_H
