//
// Created by serik1987 on 12.01.2020.
//

#define _USE_MATH_DEFINES
#include <cmath>
#include "QuasiTimeSynchronization.h"
#include "../tracereading/TraceReader.h"


namespace GLOBAL_NAMESPACE {

    QuasiTimeSynchronization::QuasiTimeSynchronization(StreamFileTrain &train) : Synchronization(train) {
        stimulusPeriod = 100.0;
        initialCycle = -1;
        finalCycle = -1;
        totalCycles = -1;
        nFramesCycle = 0.0;
    }

    QuasiTimeSynchronization::QuasiTimeSynchronization(QuasiTimeSynchronization &&other) noexcept:
        Synchronization(std::move(other)){

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;
        totalCycles = other.totalCycles;
        nFramesCycle = other.nFramesCycle;

    }

    QuasiTimeSynchronization &QuasiTimeSynchronization::operator=(QuasiTimeSynchronization &&other) noexcept {
        Synchronization::operator=(std::move(other));

        stimulusPeriod = other.stimulusPeriod;
        initialCycle = other.initialCycle;
        finalCycle = other.finalCycle;
        totalCycles = other.totalCycles;
        nFramesCycle = other.nFramesCycle;

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
        TraceReader reader(train);
        PixelListItem timeChannel(reader, PixelListItem::ARRIVAL_TIME, 0);
        reader.addPixel(timeChannel);
        if (progressFunction != nullptr){
            reader.setProgressFunction(progressFunction, "Synchronization", handle);
        }
        reader.read();
        int N = reader.getFrameNumber();

        const double* trace = reader.getTraces();
        std::vector<double> frameIntervals(N-1);
        for (int i=0; i < N-1; ++i){
            frameIntervals[i] = trace[i+1] - trace[i];
        }

        double average_ti = 0;
        for (double& interval: frameIntervals){
            average_ti += interval;
        }
        double average_frame_interval = average_ti / frameIntervals.size();
        nFramesCycle = stimulusPeriod / average_frame_interval;
        totalCycles = (int)floor(train.getTotalFrames() / nFramesCycle);
        if (totalCycles < 1){
            clearState();
            throw StimulusPeriodException();
        }
        if (initialCycle == -1){
            initialCycle = 1;
        }
        if (finalCycle == -1){
            finalCycle = totalCycles;
        }
        if (finalCycle > totalCycles){
            clearState();
            throw FinalCycleException();
        }
        initialFrame = (int)rint(((initialCycle - 1) * nFramesCycle));
        finalFrame = (int)rint(finalCycle * nFramesCycle) - 1;

        synchronizationPhase = new double[getFrameNumber()];
        double increment = 2 * M_PI / nFramesCycle;
        for (int i=0; i < getFrameNumber(); ++i){
            synchronizationPhase[i] = i * increment;
        }
    }

    void QuasiTimeSynchronization::calculatePhaseIncrement() {
        if (isDoPrecise()){
            phaseIncrement = 2 * M_PI / nFramesCycle / getCycleNumber();
        } else {
            phaseIncrement = 2 * M_PI / getFrameNumber();
        };
        initialPhase = 0.0;
    }


}