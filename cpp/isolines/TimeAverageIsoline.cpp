//
// Created by serik1987 on 12.01.2020.
//

#include <cmath>
#include "../synchronization/ExternalSynchronization.h"
#include "../tracereading/TraceReaderAndCleaner.h"
#include "TimeAverageIsoline.h"

namespace GLOBAL_NAMESPACE {

    TimeAverageIsoline::TimeAverageIsoline(StreamFileTrain &train, Synchronization &sync) : Isoline(train, sync) {
        averageCycles = 1;
    }

    TimeAverageIsoline::TimeAverageIsoline(const TimeAverageIsoline &other): Isoline(other) {
        averageCycles = other.averageCycles;
    }

    TimeAverageIsoline &TimeAverageIsoline::operator=(const TimeAverageIsoline &other) {
        Isoline::operator=(other);
        averageCycles = other.averageCycles;

        return *this;
    }

    void TimeAverageIsoline::setAverageCycles(int r) {
        if (r > 0){
            averageCycles = r;
        } else {
            throw AverageCyclesException();
        }
    }

    void TimeAverageIsoline::printSpecial(std::ostream &out) const {
        out << "Average radius, cycles: " << getAverageCycles() << "\n";
    }

    void TimeAverageIsoline::extendRange() {
        isolineInitialCycle = sync().getInitialCycle();
        if (isolineInitialCycle != -1){
            isolineInitialCycle -= averageCycles;
            sync().setInitialCycle(isolineInitialCycle);
        }
        isolineFinalCycle = sync().getFinalCycle();
        if (isolineFinalCycle != -1){
            isolineFinalCycle += averageCycles;
            sync().setFinalCycle(isolineFinalCycle);
        }
    }

    void TimeAverageIsoline::sacrifice() {
        if (sync().getCycleNumber() <= 2 * averageCycles){
            throw ExternalSynchronization::TooFewFramesException();
        }
        std::cout << isolineInitialCycle + averageCycles << std::endl;
        sync().setInitialCycle(isolineInitialCycle + averageCycles);
        sync().setFinalCycle(isolineFinalCycle - averageCycles);
    }

    void TimeAverageIsoline::traceCleaning(TraceReaderAndCleaner &reader, const double *src, const double *srcLeft,
                                           const double *srcRight, double *isolines) {

        int chans = reader.getChannelNumber();
        int frames = reader.getFrameNumber();
        int cycles = sync().getCycleNumber();
        int radius = averageCycles * (int)rint((double)(frames + 1) / cycles); // to be expressed in frames

        const double* srcIt = src;
        double* isolineIt = isolines;

        for (int frame = 0; frame < frames; ++frame){
            const double* first = srcIt - chans * (radius - 1);
            if (first < srcLeft) first = srcLeft;
            const double* last = srcIt + chans * radius;
            if (last > srcRight) last = srcRight;
            int timeinterval_local = (int)(last - first) / chans;
            for (int chan=0; chan < chans; ++chan){
                if (reader.getPixelItem(chan).getPointType() == PixelListItem::PixelValue) {
                    int counter = 0;
                    for (const double *avgIt = first; avgIt < last; avgIt += chans) {
                        isolineIt[chan] += avgIt[chan];
                        counter++;
                    }
                    isolineIt[chan] /= timeinterval_local;
                }
            }

            srcIt += chans;
            isolineIt += chans;

            removed = true;
            if (progressFunction != nullptr && frame % 100 == 0){
                int status = progressFunction(frame, frames, "Trace cleaning", progressHandle);
                if (!status){
                    removed = false;
                    clearState();
                    break;
                }
            }
        }

    }

    void TimeAverageIsoline::initialize(Accumulator &accumulator) {
        std::cout << "TIME AVERAGE INITIALIZATION\n";
    }
}