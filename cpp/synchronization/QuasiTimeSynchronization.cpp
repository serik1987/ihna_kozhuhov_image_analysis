//
// Created by serik1987 on 12.01.2020.
//

#include "QuasiTimeSynchronization.h"


namespace GLOBAL_NAMESPACE {

    QuasiTimeSynchronization::QuasiTimeSynchronization(StreamFileTrain &train) : Synchronization(train) {
        stimulusPeriod = 100.0;
        initialCycle = -1;
        finalCycle = -1;
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

    void QuasiTimeSynchronization::setStimulusPeriod(double period) {
        if (period > 0){
            stimulusPeriod = period;
        } else {
            throw StimulusPeriodException();
        }
    }

    void QuasiTimeSynchronization::setInitialCycle(int n) {
        bool assert = false;

        if (finalCycle == -1){
            assert = n > 0;
        } else {
            assert = n > 0 && n <= finalCycle;
        }

        if (assert){
            initialCycle = n;
        } else {
            throw InitialCycleException();
        }
    }

    void QuasiTimeSynchronization::setFinalCycle(int n) {
        bool assert = false;

        if (initialCycle == -1){
            assert = n > 0;
        } else {
            assert = n >= initialCycle;
        }

        if (assert){
            finalCycle = n;
        } else {
            throw FinalCycleException();
        }
    }

    void QuasiTimeSynchronization::specialPrint(std::ostream &out) const {
        out << "Stimulus period: " << getStimulusPeriod() << "\n";
        out << "Initial cycle: " << getInitialCycle() << "\n";
        out << "Final cycle: " << getFinalCycle() << "\n";
    }

    void QuasiTimeSynchronization::calculateSynchronizationPhase() {
        printf("Quasi-time synchronization: calculate the synchronization phase\n");
    }


}