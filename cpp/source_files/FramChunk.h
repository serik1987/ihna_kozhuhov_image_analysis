//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMCHUNK_H

#include "Chunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * The chunk contains general information about the certain frame.
     * The properties are applicable for continuous as well as for episodic stimulation
     *
     * The chunk shall be presented at the beginning of each frame. This is the first and only
     * the first chunk within each frame
     */
    class FramChunk: public Chunk {
    public:

        static constexpr double BIG_LONG = 4294967296.0;

#pragma pack(push, 1)
        struct FRAM_CHUNK{		// Experiment independent chunk sizeof(FRAM_CHUNK)=64
            char		Tag[4] = "\x00\x00\x00";			//01 01
            uint32_t		FrameSeqNumber = 0;	        //02 02 Frame sequential number as reported by frame grabber
            uint32_t		FrameRingNumber = 0;	//03 03 Frame ring number as reported by frame grabber
            uint32_t		TimeArrivalUsecLo = 0;	//04 04 Arrival time(microsec) as reported by frame grabber, low part
            uint32_t		TimeArrivalUsecHi = 0;	//05 05 Arrival time(microsec) as reported by frame grabber, high part
            uint32_t		TimeDelayUsec = 0;		//06 06 Difference between arrival and WaitEvent time
            uint32_t		PotentiallyBad = 0;		//07 07 As returned by GrabWaitFrameEx
            uint32_t		Locked = 0;			//08 08 As returned by GrabWaitFrameEx
            uint32_t		FrameSeqCount = 0;		//09 09 Frame sequential number (should be: FrameSeqCount+1=FrameSeqNumber)
            uint32_t		CallbackResult = 0;		//10 10 Return value of the experiment callback
            uint32_t		Free[4] = {0, 0, 0, 0};		//11 14
        };
#pragma pack(pop)

    private:
        FRAM_CHUNK info = {};

    public:
        /**
         * Initializes the chunk
         *
         * @param size actual chunk size present in the file
         */
        explicit FramChunk(uint32_t size): Chunk("FRAM", size){
            body = (char*)&info;
        }

        /**
         *
         * @return the frame tag (Please, be attentive! The string is not NULL-terminated!)
         */
        [[nodiscard]] const char* getTag() const { return info.Tag; }

        /**
         *
         * @return frame sequential number as reported by frame grabber
         */
        [[nodiscard]] uint32_t getFrameSequentialNumber() const { return info.FrameSeqNumber; }

        /**
         *
         * @return frame ring number as reported by frame grabber
         */
        [[nodiscard]] uint32_t getFrameRingNumber() const { return info.FrameRingNumber; }

        /**
         *
         * @return exact time of the frame registration in usec (the lower part)
         */
        [[nodiscard]] uint32_t getTimeArrivalUsecLo() const { return info.TimeArrivalUsecLo; }

        /**
         *
         * @return exact time of the frame registration in usec (the higher part)
         */
        [[nodiscard]] uint32_t getTimeArrivalUserHi() const { return info.TimeArrivalUsecHi; }

        /**
         *
         * @return frame arrival time in ms
         */
        [[nodiscard]] double getTimeArrival() const {
            return ((double)info.TimeArrivalUsecLo + BIG_LONG * (double)info.TimeArrivalUsecHi) * 1e-3;
        }

        /**
         *
         * @return Difference between arrival and WaitEvent time in usec
         */
        [[nodiscard]] uint32_t getTimeDelayUsec() const { return info.TimeDelayUsec; }

        /**
         *
         * @return As returned by GrabWaitFrameEx
         */
        [[nodiscard]] uint32_t getPotentiallyBad() const { return info.PotentiallyBad; }


        /**
         *
         * @return As returned by GrabWaitFrameEx
         */
        [[nodiscard]] uint32_t getLocked() const { return info.Locked; }

        /**
         *
         * @return Frame sequential number (should be: FrameSeqCount+1=FrameSeqNumber)
         */
        [[nodiscard]] uint32_t getFrameSeqCount() const { return info.FrameSeqCount; }

        /**
         *
         * @return Return value of the experiment callback
         */
        [[nodiscard]] uint32_t getCallbackResult() const { return info.CallbackResult; }

        friend std::ostream& operator<<(std::ostream& out, const FramChunk& chunk);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_FRAMCHUNK_H
