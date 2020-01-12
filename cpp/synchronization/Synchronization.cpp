//
// Created by serik1987 on 12.01.2020.
//

#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    Synchronization::Synchronization(StreamFileTrain &train): train(train) {
        doPrecise = false;
        synchronized = false;
        referenceSignalCos = nullptr;
        referenceSignalSin = nullptr;
        harmonic = 1.0;
        initialFrame = -1;
        finalFrame = -1;
        synchronizationPhase = nullptr;
        phaseIncrement = 0.0;
        initialPhase = 0.0;
    }

    Synchronization::~Synchronization(){
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE SYNCHRONIZATION\n";
#endif
        delete [] referenceSignalCos;
        delete [] referenceSignalSin;
        delete [] synchronizationPhase;
    }

    Synchronization::Synchronization(Synchronization&& other) noexcept: train(other.train){
        doPrecise = other.doPrecise;
        synchronized = other.synchronized;
        referenceSignalCos = other.referenceSignalCos;
        other.referenceSignalCos = nullptr;
        referenceSignalSin = other.referenceSignalSin;
        other.referenceSignalSin = nullptr;
        harmonic = other.harmonic;
        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;
        synchronizationPhase = other.synchronizationPhase;
        other.synchronizationPhase = nullptr;
        phaseIncrement = other.phaseIncrement;
        initialPhase = other.initialPhase;
    }

    Synchronization& Synchronization::operator=(Synchronization&& other) noexcept{
        doPrecise = other.doPrecise;
        synchronized = other.synchronized;
        referenceSignalCos = other.referenceSignalCos;
        other.referenceSignalCos = nullptr;
        referenceSignalSin = other.referenceSignalSin;
        other.referenceSignalSin = nullptr;
        harmonic = other.harmonic;

        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;
        train = other.train;
        synchronizationPhase = other.synchronizationPhase;
        other.synchronizationPhase = nullptr;
        phaseIncrement = other.phaseIncrement;
        initialPhase = other.initialPhase;

        return *this;
    }
}