//
// Created by serik1987 on 12.01.2020.
//

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
}