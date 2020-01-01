//
// Created by serik1987 on 01.01.2020.
//

#include "Frame.h"

namespace GLOBAL_NAMESPACE{

    Frame::Frame(const FileTrain& train){
        parent = &train;
        iLock = false;
        framChunk = nullptr;
        stimulationChunk = nullptr;
        emode = train.getExperimentalMode();
        body = new uint16_t[train.getXYSize()];
    }

    Frame::~Frame(){
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE FRAME\n";
#endif
        delete [] body;
    }

    std::ostream &operator<<(std::ostream &out, const Frame &frame) {
        out << "===== FRAME # " << frame.getFrameNumber() << " =====\n";
        if (frame.iLock){
            out << "The frame is locked\n";
        }
        if (frame.emode == FileTrain::Continuous){
            out << "Continuous experiment mode\n";
        } else if (frame.emode == FileTrain::Episodic){
            out << "Episodic experiment mode\n";
        }
        if (frame.framChunk == nullptr){
            out << "No frame was loaded\n";
        } else {
            out << *frame.framChunk << "\n";
        }
        if (frame.stimulationChunk != nullptr){
            out << *frame.stimulationChunk;
        }

        return out;
    }

}
