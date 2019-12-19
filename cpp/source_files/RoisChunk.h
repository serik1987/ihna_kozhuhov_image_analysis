//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_ROISCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_ROISCHUNK_H

#include "Chunk.h"

namespace ihna::kozhukhov::image_analysis {

    /**
     * Actually, we don't have any idea about why does this chunk need
     */
    class RoisChunk: public Chunk {
    public:
#pragma PACK(push, 1)
        struct ROIS_CHUNK{
            char		Tag[4] = "\x00\x00\x00";		//
            uint32_t		RecordSize = 0;	//
            uint32_t		RecordCount = 0;	//
        };
#pragma PACK(pop)

    private:
        ROIS_CHUNK info;

    public:
        explicit RoisChunk(uint32_t size): Chunk("ROIS", size) {
            body = (char*)&info;
        }

        [[nodiscard]] const char* getChunkTag() const { return info.Tag; }
        [[nodiscard]] uint32_t getRecordSize() const { return info.RecordSize; }
        [[nodiscard]] uint32_t getRecordCount() const { return info.RecordCount; }

    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_ROISCHUNK_H
