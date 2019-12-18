//
// Created by serik1987 on 16.12.2019.
//

#include "CostChunk.h"

namespace iman{

    std::ostream &operator<<(std::ostream &out, const CostChunk &chunk) {
        out << "===== COST =====\n";
        out << "Number of active synchronization channels: " << chunk.getSynchonizationChannels() << "\n";
        for (int i=0; i < chunk.getSynchonizationChannels(); ++i){
            out << "Max. value on synchronization channel " << i << ": " << chunk.getSynchronizationChannelsMax(i)
            << "\n";
        }
        out << "Number of stimulus channels: " << chunk.getStimulusChannel() << "\n";
        for (int i=0; i < chunk.getStimulusChannel(); ++i){
            out << "Stimulus period for stimulus channel " << i << ", ms: " << chunk.getStimulusPeriod(i) << "\n";
        }
        out << "Indices of stimulus channels don't coincide with indices of synchronization channels";

        return out;
    }
}