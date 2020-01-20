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
        progressFunction = nullptr;
        handle = nullptr;

        if (!train.isOpened()){
            throw FileNotOpenedException();
        }
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
        progressFunction = other.progressFunction;
        handle = other.handle;
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

        progressFunction = other.progressFunction;
        handle = other.handle;

        return *this;
    }

    const double *Synchronization::getSynchronizationPhase() const {
        if (!synchronized || synchronizationPhase  == nullptr){
            throw NotSynchronizedException();
        } else {
            return synchronizationPhase;
        }
    }

    const double *Synchronization::getReferenceSignalCos() const {
        if (!synchronized || referenceSignalCos == nullptr){
            throw NotSynchronizedException();
        } else {
            return referenceSignalCos;
        }
    }

    const double *Synchronization::getReferenceSignalSin() const {
        if (!synchronized || referenceSignalSin == nullptr){
            throw NotSynchronizedException();
        } else {
            return referenceSignalSin;
        }
    }

    std::ostream &operator<<(std::ostream &out, const Synchronization &sync) {
        out << "===== SYNCHRONIZATION =====\n";
        out << "Synchronization type: " << sync.getName() << "\n";
        out << "Initial frame: " << sync.getInitialFrame() << "\n";
        out << "Final frame: " << sync.getFinalFrame() << "\n";
        out << "Frame number: " << sync.getFrameNumber() << "\n";
        if (sync.isDoPrecise()) {
            out << "Precise analysis is ON\n";
        } else {
            out << "Precise analysis is OFF\n";
        }
        out << "Harmonic: " << sync.getHarmonic() << "\n";
        if (sync.isSynchronized()){
            out << "Synchronization is completed\n";
            out << "Phase increment, rad: " << sync.getPhaseIncrement() << "\n";
            out << "Initial phase, rad: " << sync.getInitialPhase() << "\n";

        } else {
            out << "Synchronization is not completed\n";
        }

        sync.specialPrint(out);

        return out;
    }

    void Synchronization::synchronize() {
        printf("\n");
        clearState();
        calculateSynchronizationPhase();

        printf("C++: Synchronization started\n");
        printf("\n");

        synchronized = true;
    }

    void Synchronization::clearState() {
        synchronized = false;
        delete [] referenceSignalCos;
        delete [] referenceSignalSin;
        delete [] synchronizationPhase;
    }
}