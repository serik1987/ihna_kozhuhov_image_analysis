//
// Created by serik1987 on 12.01.2020.
//

#include "QuasiStimulusSynchronization.h"

namespace GLOBAL_NAMESPACE {

    QuasiStimulusSynchronization::QuasiStimulusSynchronization(StreamFileTrain &other) : Synchronization(other) {
        stimulusPeriod = 1;
        initialCycle = -1;
        finalCycle = -1;
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

    void QuasiStimulusSynchronization::setStimulusPeriod(int period){
        if (period > 0 && period <= train.getTotalFrames()){
            stimulusPeriod = period;
        } else {
            throw StimulusPeriodException();
        }
    }

    void QuasiStimulusSynchronization::setInitialCycle(int n) {
        bool assert = false;
        if (finalCycle == -1){
            assert = n > 0 && n <= train.getTotalFrames() / stimulusPeriod;
        } else {
            assert = n > 0 && n <= finalCycle;
        }

        if (assert){
            initialCycle = n;
        } else {
            throw InitialCycleException();
        }
    }

    void QuasiStimulusSynchronization::setFinalCycle(int n){
        bool assert = false;
        if (initialCycle == -1){
            assert = n > 0 && n <= train.getTotalFrames() / stimulusPeriod;
        } else {
            assert = n >= initialCycle && n <= train.getTotalFrames() / stimulusPeriod;
        }

        if (assert){
            finalCycle = n;
        } else {
            throw FinalCycleException();
        }
    }


}