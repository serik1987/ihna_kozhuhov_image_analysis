//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H

#include "Chunk.h"

namespace iman {

    /**
     * The chunk which body contains sequence of frames that goes one to another without
     * any gap. All frames shall be the same size and the same chunk sequence.
     * Each frame contains frame header (several chunks that are specific for frame) and
     * frame body (a 2D matrix representing a certain image).
     */
    class DataChunk: public Chunk {
    public:
        /**
         * Initializes the DATA chunk
         *
         * @param size chunk size
         */
        explicit DataChunk(uint32_t size): Chunk("DATA", size) {
            body = nullptr;
        };
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H
