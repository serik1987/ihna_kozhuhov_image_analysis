//
// Created by serik1987 on 12.01.2020.
//

#include "ExternalSynchronization.h"

namespace GLOBAL_NAMESPACE {

    ExternalSynchronization::ExternalSynchronization(StreamFileTrain &train) : Synchronization(train) {
        synchronizationChannel = 0;
        synchronizationSignal = nullptr;
        initialCycle = 1;
        finalCycle = 1;
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
}