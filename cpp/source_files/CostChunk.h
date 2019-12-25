//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_COSTCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_COSTCHUNK_H

#include "../../init.h"

#include "Chunk.h"


namespace GLOBAL_NAMESPACE {


    class CostChunk: public Chunk {
    public:
#pragma PACK(push, 1)
        struct COST_CHUNK{
            char		    Tag[4] = "\x00\x00\x00";			//01 01
            uint32_t		NSynchChannels = 0;		//02 02 Up to 4 channels. Add new values bellow if needed or use SYNC_CHUNK
            uint32_t		SynchChannelMax[4] = {0, 0, 0, 0};	//03 06
            uint32_t		NStimulusChanels = 0;	//07 07 Used if no Sync Channels present
            uint32_t		StimulusPeriod[4] = {0, 0, 0, 0};	//08 11 milliseconds
            char		    Comments[20] = "\x00";		//12 16
        };
#pragma PACK(pop)

    private:
        COST_CHUNK info;

    public:
        explicit CostChunk(uint32_t size): Chunk("COST", size){
            body = (char*)&info;
        }

        /**
         *
         * @return chunk tag
         */
        [[nodiscard]] const char* getTag() const { return info.Tag; }

        /**
         *
         * @return number of synchronization channels
         */
        [[nodiscard]] int getSynchronizationChannels() const { return info.NSynchChannels; }

        /**
         *
         * @param chan number of the synchronization channel
         * @return max. value of the signal in this synchronization channel
         */
        [[nodiscard]] uint32_t getSynchronizationChannelsMax(int chan) const { return info.SynchChannelMax[chan]; }

        /**
         *
         * @return number of stimulus channels (if no synch channels present)
         */
        [[nodiscard]] int getStimulusChannel() const { return info.NStimulusChanels; }

        /**
         *
         * @param chan channel number
         * @return stimulus period in ms
         */
        [[nodiscard]] uint32_t getStimulusPeriod(int chan) const { return info.StimulusPeriod[chan]; }

        friend std::ostream& operator<<(std::ostream& out, const CostChunk& chunk);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_COSTCHUNK_H
