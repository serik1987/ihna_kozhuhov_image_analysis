//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H

#include "Isoline.h"

namespace GLOBAL_NAMESPACE {

    class TimeAverageIsoline: public Isoline {
    private:
        int averageCycles;

    public:
        TimeAverageIsoline(StreamFileTrain& train, Synchronization& sync);
        TimeAverageIsoline(const TimeAverageIsoline& other);

        TimeAverageIsoline& operator=(const TimeAverageIsoline& other);

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H
