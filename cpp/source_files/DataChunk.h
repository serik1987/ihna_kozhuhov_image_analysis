//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H

#include "Chunk.h"

namespace ihna::kozhukhov::image_analysis {

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

        class data_chunk_read_exception: public iman_exception{
        public:
            data_chunk_read_exception():
                iman_exception("The readFromFile function allows you to read the data from file header or file footer."
                               " Its function is not to read the data from the file body. Since the DATA chunk "
                               "corresponds to the file body, call DataChunk::readFromFile is considered to be an "
                               "error itself") {};
        };

        void readFromFile(SourceFile& file) override { throw data_chunk_read_exception(); };

        friend std::ostream& operator<<(std::ostream& out, const DataChunk& chunk);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_DATACHUNK_H
