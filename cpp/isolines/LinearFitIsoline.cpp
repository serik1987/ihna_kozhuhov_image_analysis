//
// Created by serik1987 on 12.01.2020.
//

#include "../tracereading/TraceReaderAndCleaner.h"
#include "LinearFitIsoline.h"
#include "../accumulators/Accumulator.h"
#include "../source_files/Frame.h"

namespace GLOBAL_NAMESPACE {

    void LinearFitIsoline::traceCleaning(TraceReaderAndCleaner &reader, const double *src, const double *srcLeft,
                                         const double *srcRight, double *isolines) {
        int chans = reader.getChannelNumber();
        int frames = reader.getFrameNumber();
        LinearFit fit(chans);
        const double* it = src;
        for (int i = 0; i < frames; ++i){
            fit.add(it);
            it += chans;
            if (it >= srcRight){
                break;
            }
        }
        fit.ready();

        double* isolinesIt = isolines;
        for (int frame = 0; frame < frames; ++frame){
            for (int chan = 0; chan < chans; ++chan){
                auto ptype = reader.getPixelItem(chan).getPointType();
                if (ptype == PixelListItem::PixelValue){
                    *isolinesIt = fit.getIntersect(chan) + fit.getSlope(chan) * frame;
                }
                ++isolinesIt;
            }
        }

        removed = true;
    }

    void LinearFitIsoline::initialize(Accumulator &accumulator) {
        linearFit = new LinearFit(accumulator.getChannelNumber());
        int frameNumber, timestamp;
        int totalFrames = getAnalysisFinalFrame() - getAnalysisInitialFrame() + 1;
        for (frameNumber = getAnalysisInitialFrame(), timestamp = 0;
                frameNumber <= getAnalysisFinalFrame(); ++frameNumber, ++timestamp){

            accumulator.readFrameData(frameNumber);
            linearFit->add(accumulator.getReadingBuffer());

            if (progressFunction != nullptr && timestamp % 100 == 0){
                bool status = progressFunction(timestamp, totalFrames, "Initialization of linear fit", progressHandle);
                if (!status){
                    throw Accumulator::InterruptedException();
                }
            }

        }
        linearFit->ready();
    }

    void LinearFitIsoline::clearState() {
        Isoline::clearState();
        delete linearFit;
        linearFit = nullptr;
    }

    void LinearFitIsoline::advance(Accumulator &accumulator, int frameNumber) {
        double* readingBuffer = accumulator.getReadingBuffer();
        for (int i=0; i < accumulator.getChannelNumber(); ++i){
            int timestamp = frameNumber - getAnalysisInitialFrame();
            readingBuffer[i] -= linearFit->getIntersect(i) + linearFit->getSlope(i) * timestamp;
        }
    }
}