//
// Created by serik1987 on 12.01.2020.
//

#include <cmath>
#include "../tracereading/TraceReader.h"
#include "../source_files/IsoiChunk.h"
#include "../source_files/CostChunk.h"
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
        getSynchronizationSignal();
        checkSynchronizationSignal();
        setFrameRange();
        setSynchronizationPhase();
    }

    void ExternalSynchronization::getSynchronizationSignal() {
        TraceReader reader(train);
        PixelListItem sync_channel(reader, PixelListItem::SYNCH, synchronizationChannel);
        reader.addPixel(sync_channel);
        if (progressFunction != nullptr){
            reader.setProgressFunction(progressFunction, "Synchronization", handle);
        }
        reader.read();

        synchronizationSignal = new double[reader.getFrameNumber()];
        for (int i=0; i < reader.getFrameNumber(); ++i){
            synchronizationSignal[i] = reader.getTraces()[i];
        }

        auto* chunk =
                dynamic_cast<CostChunk*>((*train.begin())->getIsoiChunk().getChunkById(ChunkHeader::COST_CHUNK_CODE));
        if (chunk == nullptr){
            clearState();
            throw FileTrain::experiment_mode_exception(&train);
        }
        synchMax = chunk->getSynchronizationChannelsMax(synchronizationChannel);

        syncFrames = reader.getFrameNumber();
    }

    void ExternalSynchronization::checkSynchronizationSignal() {
        double threshold = 0.5 * synchMax;
        double first = synchronizationSignal[0];
        double current = synchronizationSignal[1];
        int current_idx = 1;

        while (fabs(current - first) < 1e-12){
            if (current_idx == syncFrames-1){
                clearState();
                throw NoSignalException();
            }
            current = synchronizationSignal[++current_idx];
        }
        double diff = current - first;

        if (diff > threshold || (diff < 0 && diff > -threshold)){
            double iDum = synchMax - 1;
            for (int i=0; i < syncFrames; ++i){
                synchronizationSignal[i] = iDum - synchronizationSignal[i];
            }
        }
    }

    int ExternalSynchronization::getNextCycle(int idx) {
        if (idx < 0){
            throw std::runtime_error("Bad usage of getNextCycle");
        }

        for (int i = idx + 1; i < syncFrames; ++i){
            if (synchronizationSignal[i]  < synchronizationSignal[i-1]){
                return i;
            }
        }

        return -1;
    }

    void ExternalSynchronization::setFrameRange() {
        int idx = getNextCycle(0);
        if (idx == -1){
            clearState();
            throw TooFewFramesException();
        }

        if (initialCycle == -1){
            initialCycle = 1;
        }
        for (int i=0; i < initialCycle-1; ++i){
            idx = getNextCycle(idx);
            if (idx == -1){
                clearState();
                throw InitialCycleException();
            }
        }
        if (getNextCycle(idx) == -1){
            throw InitialCycleException();
        }
        initialFrame = idx;

        if (finalCycle != -1) {
            int ncycles = finalCycle - initialCycle + 1;
            for (int i=0; i < ncycles; ++i){
                idx = getNextCycle(idx);
                if (idx == -1){
                    throw FinalCycleException();
                }
            }
        } else {
            int ncycles = -1;
            int prev_idx = idx;
            while (idx != -1){
                prev_idx = idx;
                idx = getNextCycle(idx);
                ++ncycles;
            }
            if (ncycles == 0){
                clearState();
                throw TooFewFramesException();
            }
            idx = prev_idx;
            finalCycle = initialCycle + ncycles - 1;
        }
        finalFrame = idx-1;
    }

    void ExternalSynchronization::setSynchronizationPhase() {
        synchronizationPhase = new double[getFrameNumber() + 2];
        synchronizationPhase[0] = 2 * M_PI * synchronizationSignal[initialFrame] / synchMax;
        int timestamp = 1;
        int cycle = 0;
        int frame = initialFrame + 1;
        for (; frame <= finalFrame; ++frame, ++timestamp){
            if (synchronizationSignal[frame] < synchronizationSignal[frame-1]){
                cycle++;
            }
            synchronizationPhase[timestamp] = 2 * M_PI * (synchronizationSignal[frame] + cycle * synchMax) / synchMax;
        }
    }
}