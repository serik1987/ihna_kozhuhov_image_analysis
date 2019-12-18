//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_COMPCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_COMPCHUNK_H

#include "Chunk.h"

namespace iman {

    /**
     * This chunk is presented in compressed files only. Such a chunk
     *
     * Compression rules:
     * The first frame remains to be the same
     * The following frames in the compressed file are differences between these frames in original file
     * and the previous frames in the original file truncated to 1 byte. If such difference for a certain pixel
     * doesn't exceed 256 such information is enough. This is not the case when such a difference is higher than
     * 256. In this case The information about a certain pixel in the map is accompanied by so called "extra pixels"
     * presented at the end of the frame.
     */
    class CompChunk: public Chunk {
    public:
#pragma PACK(push, 1)
        struct COMP_CHUNK{
            char		    Tag[4] = "\x00\x00\x00";			//01 01
            uint32_t		CompressedRecordSize = 4;	//02 02
            uint32_t		CompressedFrameSize = 0;	//03 03
            uint32_t		CompressedFrameNumber = 0;	//04 04
            uint32_t	    Free[3] = {0, 0, 0};		//05 07
        };
#pragma PACK(pop)

    private:
        COMP_CHUNK info;

    public:
        explicit CompChunk(uint32_t size): Chunk("COMP", size) {
            body = (char*)&info;
        }

        /**
         *
         * @return the chunk tag
         */
        [[nodiscard]] const char* getTag() const { return info.Tag; }

        /**
         *
         * @return size of the single extrapixel
         */
        [[nodiscard]] uint32_t getCompressedRecordSize() const { return info.CompressedRecordSize; }

        /**
         *
         * @return frame size when this is in compressed state
         */
        [[nodiscard]] uint32_t getCompressedFrameSize() const { return info.CompressedFrameSize; }

        /**
         *
         * @return frame number in the compressed file
         */
        [[nodiscard]] uint32_t getCompressedFrameNumber() const { return info.CompressedFrameNumber; }

        friend std::ostream& operator<<(std::ostream& out, const CompChunk& chunk);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_COMPCHUNK_H
