//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_GREENCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_GREENCHUNK_H

#include "Chunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This chunk contains properties specific for so called "green image"
     */
    class GreenChunk: public Chunk {
    public:
#pragma pack(push, 1)
        struct GREE_CHUNK{
            char		Tag[4] = "\x00\x00\x00";			//01 01
            float 	LoClip = 0;			//02 02
            float		HiClip = 0;			//03 03
            uint32_t		LoPass = 0;			//04 04
            uint32_t		HiPass = 0;   		//05 05
            char		Comments[12] = "";		//06 08
        };
#pragma pack(pop)

    private:
        GREE_CHUNK info;

    public:
        explicit GreenChunk(uint32_t size): Chunk("GREE", size) {
            body = (char*)&info;
        }

        [[nodiscard]] const char* getTag() const { return info.Tag; }
        [[nodiscard]] float getLoClip() const { return info.LoClip; }
        [[nodiscard]] float getHiClip() const { return info.HiClip; }
        [[nodiscard]] uint32_t getLoPass() const { return info.LoPass; }
        [[nodiscard]] uint32_t getHiPass() const { return info.HiPass; }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_GREENCHUNK_H
