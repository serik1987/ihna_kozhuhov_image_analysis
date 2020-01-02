//
// Created by serik1987 on 16.12.2019.
//

#include "FramCostChunk.h"

namespace GLOBAL_NAMESPACE{

    std::ostream &operator<<(std::ostream &out, const FramCostChunk &chunk) {
        out << "===== cost =====\n";
        out << "Size of the remaining fields: " << chunk.getFrameHeaderSize();
        for (int idx = 0; idx < 4; ++idx){
            out << "\n" << "Value from synchronization channel " << idx << ": " << chunk.getSynchChannel(idx);
            out << "\n" << "Delay from synchronization channel " << idx << ": " << chunk.getSynchChannelDelay(idx);
        }

        return out;
    }
}