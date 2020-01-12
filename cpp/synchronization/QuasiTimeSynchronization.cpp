//
// Created by serik1987 on 12.01.2020.
//

#include "QuasiTimeSynchronization.h"


namespace GLOBAL_NAMESPACE {

    QuasiTimeSynchronization::QuasiTimeSynchronization(StreamFileTrain &train) : Synchronization(train) {
        stimulusPeriod = 1.0;
        initialCycle = 1;
        finalCycle = 1;
    }

    QuasiTimeSynchronization::QuasiTimeSynchronization(QuasiTimeSynchronization &&other) noexcept:
        Synchronization(std::move(other)){

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;

    }

    QuasiTimeSynchronization &QuasiTimeSynchronization::operator=(QuasiTimeSynchronization &&other) noexcept {
        Synchronization::operator=(std::move(other));

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;

        return *this;
    }


}