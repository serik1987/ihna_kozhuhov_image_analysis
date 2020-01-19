//
// Created by serik1987 on 12.01.2020.
//

#include "ExternalSynchronization.h"

namespace GLOBAL_NAMESPACE {

    ExternalSynchronization::ExternalSynchronization(StreamFileTrain &train) : Synchronization(train) {
        synchronizationChannel = 0;
        synchronizationSignal = nullptr;
        initialCycle = -1;
        finalCycle = -1;

        if (train.getExperimentalMode() != FileTrain::Continuous){
            throw FileTrain::experiment_mode_exception(&train);
        }
    }

    ExternalSynchronization::ExternalSynchronization(ExternalSynchronization &&other) noexcept:
        Synchronization(std::move(other)){

        synchronizationChannel = other.synchronizationChannel;
        synchronizationSignal = other.synchronizationSignal;
        other.synchronizationSignal = nullptr;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;

    }

    ExternalSynchronization::~ExternalSynchronization() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE EXTERNAL SYNCHRONIZATION\n";
#endif
        delete [] synchronizationSignal;
    }

    ExternalSynchronization &ExternalSynchronization::operator=(ExternalSynchronization &&other) noexcept {
        Synchronization::operator=(std::move(other));

        synchronizationChannel = other.synchronizationChannel;
        synchronizationSignal = other.synchronizationSignal;
        other.synchronizationSignal = nullptr;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;

        return *this;
    }

    void ExternalSynchronization::setSynchronizationChannel(int chan) {
        if (chan >= 0 && chan < train.getSynchronizationChannelNumber()){
            synchronizationChannel = chan;
        } else {
            throw SynchronizationChannelException();
        }
    }

    void ExternalSynchronization::setInitialCycle(int n) {
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

    void ExternalSynchronization::setFinalCycle(int n) {
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

    void ExternalSynchronization::specialPrint(std::ostream &out) const {
        out << "Synchronization channel: " << getSynchronizationChannel() << "\n";
        out << "Initial cycle: " << getInitialCycle() << "\n";
        out << "Final cycle: " << getFinalCycle() << "\n";
    }

    void ExternalSynchronization::calculateSynchronizationPhase() {
        printf("C++: calculating the synchronization phase\n");
    }
}