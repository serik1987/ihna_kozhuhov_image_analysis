//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMCOSTCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMCOSTCHUNK_H

#include "Chunk.h"

namespace ihna::kozhukhov::image_analysis {

    /**
     * The chunk contains information about the frame related to continuous (sine-wave) stimulation
     *
     * If the stimulation is continuous and COST chunk is present in the header of the file, this chunk
     * presents in the header of each frame. Otherwise, the chunk is fully ignored
     * The chunk must be the second and only the second within each frame header
     */
    class FramCostChunk: public Chunk {
    public:
#pragma PACK(push, 1)
        struct FRAM_COST_CHUNK{
            char		    Tag[4] = "\x00\x00\x00";			//01 01
            uint32_t		HeaderSize = 0;		//02 02 size of the frameheader to follow(sizeof(FRAM_COST_CHUNK)-16)
            uint32_t		SynchChannel[4] = {0, 0, 0, 0};	//02 06 Up to 4 channels. Increase the arrays size if needed or use SYNC_CHUNK
            uint32_t		SynchChannelDelay[4] = {0, 0, 0, 0};	//06 10 Relative to arrival time (microseconds)
            uint32_t		Free[4] = {0, 0, 0, 0};		//10 14
        };
#pragma PACK(pop)

    private:
        FRAM_COST_CHUNK info;

    public:
        /**
         * Initializes the chunk
         *
         * @param size chunk size
         */
        explicit FramCostChunk(uint32_t size): Chunk("cost", size) {
            body = (char*)&info;
        }

        /**
         *
         * @return the chunk tag
         */
        [[nodiscard]] const char* getTag() const { return info.Tag; }

        /**
         *
         * @return size of the remaining fields
         */
        [[nodiscard]] uint32_t getFrameHeaderSize() const { return info.HeaderSize; }

        /**
         *
         * @param idx synch channel number
         * @return stimulus sample at the current synchronization channel
         */
        [[nodiscard]] uint32_t getSynchChannel(int idx) const { return info.SynchChannel[idx]; }

        /**
         *
         * @param idx synch channel number
         * @return estimated delay time at a current synchronization channel
         */
        [[nodiscard]] uint32_t getSynchChannelDelay(int idx)const { return info.SynchChannelDelay[idx]; }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMCOSTCHUNK_H
