//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMEPSTCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMEPSTCHUNK_H

#include "Chunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * The chunk relates frame information related only to the episodic stimulation
     *
     * If the data were recorded under episodic stimulation (i.e., under flashed semi-stationary
     * stimulus) and EPST chunk is presented in the file header, this chunk is presented within
     * each frame header. Otherwise, the chunk shall be ignored.
     *
     * The frame position relatively to the beginning of the frame shall be the same for all frames
     * in the record. Otherwise, There will be failure in reading the file
     */
    class FramEpstChunk: public Chunk {
    public:
#pragma PACK(push, 1)
        struct FRAM_EPST_CHUNK{
            char		Tag[4];			//01 01
            uint32_t		HeaderSize;		//02 02 size of the frameheader to follow(sizeof(FRAM_EPST_CHUNK)-16)
            uint32_t		SeqNumber;		//03 03 sequence number 0,1,2,...NOT including paused frames
            uint32_t		Repetition;		//04 04 repetition number of this cycle
            uint32_t		Trial;			//05 05 number of stim in this cycle
            uint32_t		Condition;		//06 06 stimulus number
            uint32_t		FrameOfCondition;	//07 07 which frame in relation to start of stimulus
            uint32_t		FramePaused;		//08 08 BOOL variable, marks paused frames(=1[actually !=0])
            uint32_t		FrameType;		//09 09 frame type, iti and stim for now
            uint32_t		Free[5];		//10 14
        };
#pragma PACK(pop)

    private:
        FRAM_EPST_CHUNK info;

    public:
        explicit FramEpstChunk(uint32_t size): Chunk("epst", size) {
            body = (char*)&info;
        }

        enum FrameType { ITI = 0, STIM = 1, PRE = 2, POST = 3, PAUSE = 4};

        /**
         *
         * @return the chunk tag
         */
        [[nodiscard]] const char* getTag() const { return info.Tag; }

        /**
         *
         * @return the remaining distance to the frame body
         */
        [[nodiscard]] uint32_t getHeaderSize() const { return info.HeaderSize; }

        /**
         *
         * @return sequence number: 0, 1, 2, 3, ...
         */
        [[nodiscard]] uint32_t getSequenceNumber() const { return info.SeqNumber; }

        /**
         *
         * @return repetition number on this cycle
         */
        [[nodiscard]] uint32_t getRepetition() const { return info.Repetition; }

        /**
         *
         * @return stimulus number on this cycle
         */
        [[nodiscard]] uint32_t getTrialNumber() const { return info.Trial; }

        /**
         *
         * @return stimulus number relatively to the beginning of the record
         */
        [[nodiscard]] uint32_t getConditionNumber() const { return info.Condition; }

        /**
         *
         * @return the frame number relatively to the stimulus onset
         */
        [[nodiscard]] uint32_t getRelativeFrameNumber() const { return info.FrameOfCondition; }

        /**
         *
         * @return true if the frame has status PAUSED
         */
        [[nodiscard]] bool isPausedFrame() const { return info.FramePaused != 0; }

        /**
         *
         * @return frame type (its relation to the stimulus
         */
        [[nodiscard]] FrameType getFrameType() const { return (FrameType)info.FrameType; }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMEPSTCHUNK_H
