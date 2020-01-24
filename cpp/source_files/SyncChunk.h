//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_SYNCCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_SYNCCHUNK_H

#include "Chunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * I actually don't have any idea why this chunk is needed
     */
    class SyncChunk: public Chunk {
    public:
#pragma pack(push, 1)
        struct SYNC_CHUNK {
            char		    Tag[4] = "\x00\x00\x00";		//01 01
            uint32_t		RecordSize = 0;	//02 02
            uint32_t		RecordType = 0;	//03 03 BYTE, USHORT,...
            uint32_t		RecordCount = 0;	//04 04
            uint32_t		ChannelNumber = 0;	//05 05
            uint32_t		ChannelMaxValue = 0;//06 06
            uint32_t		StimulusPeriod = 0; //07 07 milliseconds
            uint32_t		Free[3] = {0, 0, 0};        //08 10
        };
#pragma pack(pop)

    private:
        SYNC_CHUNK info;

    public:
        explicit SyncChunk(uint32_t size): Chunk("SYNC", size){
            body = (char*)&info;
        }

        [[nodiscard]] const char* getTag() const { return info.Tag; }
        [[nodiscard]] uint32_t getRecordSize() const { return info.RecordSize; }
        [[nodiscard]] uint32_t getRecordType() const { return info.RecordType; }
        [[nodiscard]] uint32_t getRecordCount() const { return info.RecordCount; }
        [[nodiscard]] uint32_t getChannelNumber() const { return info.ChannelNumber;}
        [[nodiscard]] uint32_t getChannelMaxValue() const { return info.ChannelMaxValue; }
        [[nodiscard]] uint32_t getStimulusPeriod() const { return info.StimulusPeriod; }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_SYNCCHUNK_H
