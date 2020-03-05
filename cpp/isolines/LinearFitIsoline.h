//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFITISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFITISOLINE_H

#include "Isoline.h"
#include "../misc/LinearFit.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Provides the isoline remove based on the linear regression
     */
    class LinearFitIsoline: public Isoline {
    private:
        LinearFit* linearFit; // for accumulator usage only

    protected:
        void printSpecial(std::ostream& out) const override {};

    public:
        LinearFitIsoline(StreamFileTrain& train, Synchronization& sync): Isoline(train, sync) {
            linearFit = nullptr;
        };

        ~LinearFitIsoline() override {
            delete linearFit;
        }

        void clearState() override;

        [[nodiscard]] const char* getName() const noexcept override { return "linear fit isoline"; }

        /**
         * Does nothing
         */
        void extendRange() override {};

        /**
         * Does nothing
         */
        void sacrifice() override {};

        void traceCleaning(TraceReaderAndCleaner& reader, const double* src, const double* srcLeft,
                const double* srcRight, double* isolines) override;

        void initialize(Accumulator& accumulator) override;
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFITISOLINE_H
