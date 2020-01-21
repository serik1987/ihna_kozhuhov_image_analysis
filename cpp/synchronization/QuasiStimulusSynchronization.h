//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H

#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This synchronization is based on assumption that the stimulus is continuous and its period equals to
     * integer number of frames and expresses in frame number
     * 1) The synchronization signal is the timestamp number multiplied by 2PI and divided by the stimulus period
     * 2) initial and final frame is defined in such a way as to contain integer number of cycles. The first cycle
     * is assumed to start at the beginning of the record
     */
    class QuasiStimulusSynchronization: public Synchronization {
    private:
        int stimulusPeriod;
        int initialCycle;
        int finalCycle;
        int cycleNumber;

    protected:
        void specialPrint(std::ostream& out) const override;

        void calculateSynchronizationPhase() override;
        void calculatePhaseIncrement() override;

    public:
        explicit QuasiStimulusSynchronization(StreamFileTrain& other);
        QuasiStimulusSynchronization(const QuasiStimulusSynchronization& other) = delete;
        QuasiStimulusSynchronization(QuasiStimulusSynchronization&& other) noexcept;

        QuasiStimulusSynchronization& operator=(const QuasiStimulusSynchronization& other) = delete;
        QuasiStimulusSynchronization& operator=(QuasiStimulusSynchronization&& other) noexcept;

        [[nodiscard]] const char* getName() const noexcept override {return "quasi-stimulus synchronization"; }

        /**
         *
         * @return the stimulus period
         */
        [[nodiscard]] int getStimulusPeriod() const { return stimulusPeriod; }

        /**
         *
         * @return initial cycle or -1 if the initial cycle has not been set. In this case the intitial cycle
         * will be set during the synchronization automatically in such a way as to maximize the analysis epoch
         */
        [[nodiscard]] int getInitialCycle() const { return initialCycle; }

        /**
         *
         * @return the final cycle or -1 if the final cycle has not been set. In this case the final cycle will
         * be set automatically in such a way as to maximize the analysis epoch
         */
        [[nodiscard]] int getFinalCycle() const { return finalCycle; }

        /**
         *
         * @return total number of cycles within the record
         */
        [[nodiscard]] int getCycleNumber() const override { return cycleNumber; }

        /**
         * Sets the stimulus period
         *
         * @param period the stimulus period in frames
         */
        void setStimulusPeriod(int period);

        /**
         * Sets the cycle from which the analysis starts
         *
         * @param n number of cycle from which the analysis starts
         */
        void setInitialCycle(int n);

        /**
         * Sets the cycle at which the analysis finishes
         *
         * @param n number of cycle at which analysis finishes
         */
        void setFinalCycle(int n);
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H
