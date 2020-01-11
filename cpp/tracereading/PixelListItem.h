//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PIXELLISTITEM_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PIXELLISTITEM_H

#include "../exceptions.h"

namespace GLOBAL_NAMESPACE {

    class TraceReader;

    class PixelListItem {
    public:
        enum PointType {ArrivalTime, SynchronizationChannel, PixelValue};
        static constexpr int ARRIVAL_TIME = -1;
        static constexpr int SYNCH = -2;

    private:
        int i, j;

        size_t arrivalTimeDisplacement, synchronizationChannelDisplacement, frameBodyDisplacement;
        size_t arrivalTimeSize, synchronizationChannelSize;

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PIXELLISTITEM_H
