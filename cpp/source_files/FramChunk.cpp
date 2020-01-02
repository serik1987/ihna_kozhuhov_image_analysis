//
// Created by serik1987 on 15.12.2019.
//

#include "FramChunk.h"

namespace GLOBAL_NAMESPACE{

    std::ostream& operator<<(std::ostream &out, const FramChunk &chunk) {
        out << "===== FRAM =====\n";
        out << "Frame sequential number: " << chunk.getFrameSequentialNumber() << "\n";
        out << "Frame ring number: " << chunk.getFrameRingNumber() << "\n";
        out << "Frame arrival time (ms): " << chunk.getTimeArrival() << "\n";
        out << "Time delay (usec): " << chunk.getTimeDelayUsec() << "\n";
        out << "Potentially bad: " << chunk.getPotentiallyBad() << "\n";
        out << "Locked: " << chunk.getLocked() << "\n";
        out << "Frame sequential count: " << chunk.getFrameSeqCount() << "\n";
        out << "Callback result: " << chunk.getCallbackResult();

        return out;
    }
}