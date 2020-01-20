//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASITIMESYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASITIMESYNCHRONIZATION_H

#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This synchronization is based on assumption that the stimulus period is exactly defined and
     * expressed in native units (seconds, milliseconds). Also, the frame arrival time (the 'TIME')
     * channel is also exactly defined.
     *
     * 1) The reference signal is a stimulus phase that linearly depends on the timestamp in such a way
     * as frame number which arrival time is N multiplied by the stimulus period is N multiplied by 2PI.
     *
     * 2) The record length will be selected in such a way as to contain integer number of stimulus cycle
     * Number of the first and the last cycle is defined by the user.
     */
    class QuasiTimeSynchronization: public Synchronization {
    private:
        double stimulusPeriod;
        int initialCycle;
        int finalCycle;
        int totalCycles;

    protected:
        void specialPrint(std::ostream& out) const override;

        void calculateSynchronizationPhase() override;

    public:
        explicit QuasiTimeSynchronization(StreamFileTrain& train);
        QuasiTimeSynchronization(const QuasiTimeSynchronization& other) = delete;
        QuasiTimeSynchronization(QuasiTimeSynchronization&& other) noexcept;

        QuasiTimeSynchronization& operator=(const QuasiTimeSynchronization& other) = delete;
        QuasiTimeSynchronization& operator=(QuasiTimeSynchronization&& other) noexcept;

        [[nodiscard]] const char* getName() const noexcept override { return "quasi-time synchronization"; }

        /**
         *
         * @return stimulus period in ms
         */
        [[nodiscard]] double getStimulusPeriod() const { return stimulusPeriod; }

        /**
         *
         * @return initial cycle
         * In case where the initial cycle is not set, the function will return -1 before the synchronization.
         * This means that the initial cycle will set automatically during the synchronization process in such
         * a way as to maximize the analysis epoch
         */
        [[nodiscard]] int getInitialCycle() const { return initialCycle; }

        /**
         *
         * @return final cycle
         * In case where the final cycle is not set, the function will return -1 before the synchronization.
         * This means that the final cycle will be set automatically during the synchronization process in such
         * a way as to maximize the analysis epoch
         */
        [[nodiscard]] int getFinalCycle() const { return finalCycle; }

        /**
         * Sets the stimulus period
         *
         * @param period stimulus period in ms
         */
        void setStimulusPeriod(double period);

        /**
         * Sets the initial cycle
         *
         * @param n the initial cycle
         */
        void setInitialCycle(int n);


        /**
         * Sets the final cycle
         *
         * @param n the final cycle
         */
        void setFinalCycle(int n);

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASITIMESYNCHRONIZATION_H
