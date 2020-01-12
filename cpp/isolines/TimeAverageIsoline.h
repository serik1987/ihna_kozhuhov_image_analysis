//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H

#include "Isoline.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Removes the isoline by smoothing the original signal and substracting the smoothed signal
     * from the original one.
     */
    class TimeAverageIsoline: public Isoline {
    private:
        int averageCycles;

    public:
        TimeAverageIsoline(StreamFileTrain& train, Synchronization& sync);
        TimeAverageIsoline(const TimeAverageIsoline& other);

        TimeAverageIsoline& operator=(const TimeAverageIsoline& other);

        [[nodiscard]] const char* getName() const noexcept override { return "time average"; }

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H
