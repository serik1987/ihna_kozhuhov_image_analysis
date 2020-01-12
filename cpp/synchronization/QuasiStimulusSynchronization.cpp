//
// Created by serik1987 on 12.01.2020.
//

#include "QuasiStimulusSynchronization.h"

namespace GLOBAL_NAMESPACE {

    QuasiStimulusSynchronization::QuasiStimulusSynchronization(StreamFileTrain &other) : Synchronization(other) {
        stimulusPeriod = 1;
        initialCycle = 1;
        finalCycle = 1;
    }

    QuasiStimulusSynchronization::QuasiStimulusSynchronization(QuasiStimulusSynchronization &&other) noexcept:
        Synchronization(std::move(other)){

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;

    }

    QuasiStimulusSynchronization &
    QuasiStimulusSynchronization::operator=(QuasiStimulusSynchronization &&other) noexcept {
        Synchronization::operator=(std::move(other));

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;

        return *this;
    }


}