//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOISOLINE_H

#include "Isoline.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Provides no isoline remove except declining the average
     */
    class NoIsoline: public Isoline {
    protected:
        void printSpecial(std::ostream& out) const override {};

    public:
        NoIsoline(StreamFileTrain& train, Synchronization& sync): Isoline(train, sync) {};

        /**
         * Does nothing
         */
        void extendRange() override {};

        /**
         * Does nothing
         */
        void sacrifice() override {};

        void traceCleaning(TraceReaderAndCleaner& reader, const double* src, const double* srcFirst,
                const double* srcLast, double* isolines) override {};

        [[nodiscard]] const char* getName() const noexcept override { return "no isoline"; }

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOISOLINE_H
