//
// Created by serik1987 on 12.01.2020.
//

#include <cmath>
#include "../synchronization/ExternalSynchronization.h"
#include "../tracereading/TraceReaderAndCleaner.h"
#include "../accumulators/Accumulator.h"
#include "TimeAverageIsoline.h"

namespace GLOBAL_NAMESPACE {

    TimeAverageIsoline::TimeAverageIsoline(StreamFileTrain &train, Synchronization &sync) : Isoline(train, sync) {
        averageCycles = 1;
        averageBuffer = nullptr;
        beforeSubstractBuffer = nullptr;
        bufferSize = 0;
        avgSize = 0;
    }

    TimeAverageIsoline::TimeAverageIsoline(const TimeAverageIsoline &other): Isoline(other) {
        averageCycles = other.averageCycles;
        averageBuffer = new double[other.bufferSize];
        std::memcpy(averageBuffer, other.averageBuffer, sizeof(double) * other.bufferSize);
        beforeSubstractBuffer = new double[other.bufferSize];
        std::memcpy(beforeSubstractBuffer, other.beforeSubstractBuffer, sizeof(double) * other.bufferSize);
        bufferSize = other.bufferSize;
        avgSize = other.avgSize;
    }

    TimeAverageIsoline &TimeAverageIsoline::operator=(const TimeAverageIsoline &other) {
        if (&other == this){
            return *this;
        }
        Isoline::operator=(other);
        averageCycles = other.averageCycles;

        delete [] averageBuffer;
        averageBuffer = new double[other.bufferSize];
        delete [] beforeSubstractBuffer;
        beforeSubstractBuffer = new double[other.bufferSize];
        bufferSize = other.bufferSize;
        avgSize = other.avgSize;

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

    TimeAverageIsoline::~TimeAverageIsoline() {
        delete [] averageBuffer;
        delete [] beforeSubstractBuffer;
    }

    void TimeAverageIsoline::clearState() {
        Isoline::clearState();

        delete [] averageBuffer;
        averageBuffer = nullptr;
        delete [] beforeSubstractBuffer;
        beforeSubstractBuffer = nullptr;
        bufferSize = 0;
        avgSize = 0;
    }

    void TimeAverageIsoline::initialize(Accumulator &accumulator) {
        bufferSize = accumulator.getChannelNumber();
        avgSize = 0;
        averageBuffer = new double[bufferSize];
        beforeSubstractBuffer = new double[bufferSize];
        for (int i=0; i < bufferSize; ++i){
            averageBuffer[i] = 0.0;
        }

        double* readingBuffer = accumulator.getReadingBuffer();
        int start = getIsolineInitialFrame();
        int cycles = sync().getCycleNumber();
        int frames = sync().getFrameNumber();
        avgRadius = averageCycles * (int)rint((double)(frames + 1) / cycles);
        int finish = getAnalysisInitialFrame() + avgRadius;
        int frame, timestamp;

        for (timestamp = 0, frame = start; frame < finish; ++frame, ++timestamp){

            accumulator.readFrameData(frame);
            ++avgSize;
            for (int j = 0; j < bufferSize; ++j){
                averageBuffer[j] += readingBuffer[j];
            }

            if (timestamp % 100 == 0 && progressFunction != nullptr){
                int status = progressFunction(timestamp, 2 * avgRadius, "Initializing time average", progressHandle);
                if (!status){
                    throw Accumulator::InterruptedException();
                }
            }
        }
    }

    void TimeAverageIsoline::advance(Accumulator &accumulator, int frameNumber) {
        double* readingBuffer = accumulator.getReadingBuffer();
        std::memcpy(beforeSubstractBuffer, readingBuffer, sizeof(double) * bufferSize);

        int finalframe_ta = getIsolineFinalFrame();
        int initframe_ta = getIsolineInitialFrame();

        if (frameNumber <= finalframe_ta - avgRadius){
            accumulator.readFrameData(frameNumber + avgRadius);
            avgSize += 1;
            for (int i=0; i < bufferSize; ++i){
                averageBuffer[i] += readingBuffer[i];
            }
        }

        if (frameNumber > initframe_ta + avgRadius){
            accumulator.readFrameData(frameNumber - avgRadius);
            avgSize -= 1;
            for (int i=0; i < bufferSize; ++i){
                averageBuffer[i] -= readingBuffer[i];
            }
        }

        for (int i=0; i < bufferSize; ++i){
            readingBuffer[i] = beforeSubstractBuffer[i] - averageBuffer[i] / avgSize;
        }
    }
}