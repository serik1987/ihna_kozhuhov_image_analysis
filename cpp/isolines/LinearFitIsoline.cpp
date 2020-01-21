//
// Created by serik1987 on 12.01.2020.
//

#include "../tracereading/TraceReaderAndCleaner.h"
#include "../misc/LinearFit.h"
#include "LinearFitIsoline.h"

namespace GLOBAL_NAMESPACE {

    void LinearFitIsoline::traceCleaning(TraceReaderAndCleaner &reader, const double *src, const double *srcLeft,
                                         const double *srcRight, double *isolines) {
        int chans = reader.getChannelNumber();
        int frames = reader.getFrameNumber();
        int points = chans * frames;
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
    }
}