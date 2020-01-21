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

        void printSpecial(std::ostream& out) const override;

    public:
        TimeAverageIsoline(StreamFileTrain& train, Synchronization& sync);
        TimeAverageIsoline(const TimeAverageIsoline& other);

        TimeAverageIsoline& operator=(const TimeAverageIsoline& other);

        [[nodiscard]] const char* getName() const noexcept override { return "time average"; }

        /**
         *
         * @return the smooth radius, in cycles. When NoSynchronization is used, the smooth radius is given in frames
         */
        [[nodiscard]] int getAverageCycles() const { return averageCycles; }

        /**
         * Sets number of the averaged cycles.
         *
         * @param r number of the averaged cycles.
         * For the purpose of time averaging r cycles will be taken from the left of the analyzing frame and r cycles
         * will be taken from the right of the analyzing frame
         */
        void setAverageCycles(int r);

        /**
         * Extends the synchronization range by r cycles to the left and r cycles to the right
         * Also, fills some private variables
         */
        void extendRange() override;

        void traceCleaning(TraceReaderAndCleaner& reader, const double* src, const double* srcLeft,
                const double* srcRight, double* isolines) override;

        /**
         * Adjusts boundaries from isoline plotting range to the analysis range
         */
        void sacrifice() override;

        class AverageCyclesException: public IsolineException {
        public:
            AverageCyclesException(): IsolineException("bad value of averaged cycles") {};
        };

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H
