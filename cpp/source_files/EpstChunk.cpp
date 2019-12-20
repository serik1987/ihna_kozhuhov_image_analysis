//
// Created by serik1987 on 16.12.2019.
//

#include "EpstChunk.h"

namespace GLOBAL_NAMESPACE{

    std::ostream &operator<<(std::ostream &out, const EpstChunk &chunk) {
        using std::endl;

        out << "===== EPST ======\n";
        out << "Number of conditions: " << chunk.getConditionNumber() << endl;
        out << "Number of repetitions: " << chunk.getRepetitionsNumber() << endl;
        if (chunk.getRandomize() > 0){
            out << "Stimuli were randomized\n";
        }
        out << "Number of intertrial frames: " << chunk.getItiFrames() << endl;
        out << "Number of stimulus frames: " << chunk.getStimulusFrames() << endl;
        out << "Number of prestimulus frames: " << chunk.getPrestimulusFrames() << endl;
        out << "Number of poststimulus frames: " << chunk.getPoststimulusFrames() << endl;

        return out;
    }
}