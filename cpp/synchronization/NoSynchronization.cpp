//
// Created by serik1987 on 12.01.2020.
//

#include <cmath>
#include "NoSynchronization.h"

namespace GLOBAL_NAMESPACE {

    NoSynchronization::NoSynchronization(StreamFileTrain& train): Synchronization(train) {
        initialFrame = 0;
        finalFrame = train.getTotalFrames() - 1;
    }

    NoSynchronization &NoSynchronization::operator=(NoSynchronization &&other) noexcept {
        Synchronization::operator=(std::move(other));

        return *this;
    }

    void NoSynchronization::setInitialFrame(int frame) {
        if (frame >= 0 && frame < finalFrame){
            initialFrame = frame;
        } else {
            throw FrameRangeException();
        }
    }

    void NoSynchronization::setFinalFrame(int frame) {
        if (frame > initialFrame && (unsigned int)frame < train.getTotalFrames()){
            finalFrame = frame;
        } else {
            throw FrameRangeException();
        }
    }

    void NoSynchronization::specialPrint(std::ostream &out) const {

    }

    void NoSynchronization::calculateSynchronizationPhase() {
        int n = getFrameNumber();
        synchronizationPhase = new double[n];
        for (int i=0; i < n; ++i){
            synchronizationPhase[i] = 2 * M_PI * i;
        }
    }

    void NoSynchronization::calculatePhaseIncrement() {
        initialPhase = 0.0;
        phaseIncrement = 2 * M_PI / getFrameNumber();
    }
}