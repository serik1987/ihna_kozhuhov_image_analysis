//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_EPSTCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_EPSTCHUNK_H

#include "Chunk.h"

namespace GLOBAL_NAMESPACE {

    /**
     * The chunk contains general information about the stimulus  in case
     * where stimulation is episodic (flashing stimuli and mean luminance
     * between them => on/off responses investigation
     */
    class EpstChunk: public Chunk {
    public:
#pragma pack(push, 1)
        struct EPST_CHUNK{
            char		Tag[4] = "\x00\x00\x00";			    //01 01
            uint32_t	NConditions = 0;		//02 02 Number of conditions (for episodic stimuli)
            uint32_t	NRepetitions = 0;		//03 03 Number of repetitions (for episodic stimuli)
            uint32_t	Randomize = 0;		    //04 04 (for episodic stimuli)
            uint32_t	NFramesITI = 0;		    //05 05 (for episodic stimuli)
            uint32_t	NFramesStim = 0;		//06 06 (for episodic stimuli)
            uint32_t    NFramesBlankPre = 0;    //07 07 Pre-stimulus frames (for episodic stimuli)
            uint32_t    NFramesBlankPost = 0;   //08 08 Post-stimulus frames (for episodic stimuli)
            char        Comments[32] = "";       //09 16
        };
#pragma pack(pop)

    private:
        EPST_CHUNK info;

    public:
        explicit EpstChunk(uint32_t size): Chunk("EPST", size){
            body = (char*)&info;
        }

        /**
         *
         * @return chunk tag
         */
        [[nodiscard]] const char* getTag() const { return info.Tag;}

        /**
         *
         * @return total number of stimulus conditions
         */
        [[nodiscard]] uint32_t getConditionNumber() const { return info.NConditions; }

        /**
         *
         * @return total number of stimulus repetitions
         */
        [[nodiscard]] uint32_t getRepetitionsNumber() const { return info.NRepetitions; }

        /**
         *
         * @return if randomize
         */
        [[nodiscard]] uint32_t getRandomize() const { return info.Randomize; }

        /**
         *
         * @return number of intertrial frames
         */
        [[nodiscard]] uint32_t getItiFrames() const { return info.NFramesITI; }

        /**
         *
         * @return total number of stimulus frames
         */
        [[nodiscard]] uint32_t getStimulusFrames() const { return info.NFramesStim; }

        /**
         *
         * @return number of prestimulus frames
         */
        [[nodiscard]] uint32_t getPrestimulusFrames() const { return info.NFramesBlankPre; }

        /**
         *
         * @return number of poststimulus frames
         */
        [[nodiscard]] uint32_t getPoststimulusFrames() const { return info.NFramesBlankPost; }

        friend std::ostream& operator<<(std::ostream& out, const EpstChunk& chunk);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_EPSTCHUNK_H
