//
// Created by serik1987 on 12.01.2020.
//

#include <cmath>
#include "QuasiStimulusSynchronization.h"

namespace GLOBAL_NAMESPACE {

    QuasiStimulusSynchronization::QuasiStimulusSynchronization(StreamFileTrain &train) : Synchronization(train) {
        stimulusPeriod = 1;
        initialCycle = -1;
        finalCycle = -1;
        cycleNumber = -1;
    }

    QuasiStimulusSynchronization::QuasiStimulusSynchronization(QuasiStimulusSynchronization &&other) noexcept:
        Synchronization(std::move(other)){

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;
        cycleNumber = other.cycleNumber;

    }

    QuasiStimulusSynchronization &
    QuasiStimulusSynchronization::operator=(QuasiStimulusSynchronization &&other) noexcept {
        Synchronization::operator=(std::move(other));

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;
        cycleNumber = other.cycleNumber;

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

    void QuasiStimulusSynchronization::specialPrint(std::ostream &out) const {
        out << "Stimulus period: " << getStimulusPeriod() << "\n";
        out << "Initial cycle: " << getInitialCycle() << "\n";
        out << "Final cycle: " << getFinalCycle() << "\n";
    }

    void QuasiStimulusSynchronization::calculateSynchronizationPhase() {
        cycleNumber = train.getTotalFrames() / stimulusPeriod;
        if (initialCycle == -1){
            initialCycle = 1;
        }
        if (finalCycle == -1){
            finalCycle = cycleNumber;
        }
        initialFrame = stimulusPeriod * (initialCycle - 1);
        finalFrame = stimulusPeriod * finalCycle - 1;

        int n = getFrameNumber();
        synchronizationPhase = new double[n];
        for (int i=0; i < n; ++i){
            synchronizationPhase[i] = 2 * M_PI * i / stimulusPeriod;
        }
    }

    void QuasiStimulusSynchronization::calculatePhaseIncrement() {
        initialPhase = 0.0;
        phaseIncrement = 2 * M_PI / getFrameNumber();
    }


}