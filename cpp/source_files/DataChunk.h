//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H

#include "Chunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * The chunk which body contains sequence of frames that goes one to another without
     * any gap. All frames shall be the same size and the same chunk sequence.
     * Each frame contains frame header (several chunks that are specific for frame) and
     * frame body (a 2D matrix representing a certain image).
     */
    class DataChunk: public Chunk {
    protected:
        void writeBody(std::ofstream& output) override {};

    public:
        /**
         * Initializes the DATA chunk
         *
         * @param size chunk size
         */
        explicit DataChunk(uint32_t size): Chunk("DATA", size) {
            body = nullptr;
        };

        class data_chunk_read_exception: public iman_exception{
        public:
            data_chunk_read_exception():
                iman_exception(MSG_DATA_CHUNK_NOT_READ_EXCEPTION) {};
        };

        void readFromFile(SourceFile& file) override { throw data_chunk_read_exception(); };

        friend std::ostream& operator<<(std::ostream& out, const DataChunk& chunk);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H
