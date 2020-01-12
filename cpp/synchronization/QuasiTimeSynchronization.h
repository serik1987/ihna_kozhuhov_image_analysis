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

    public:
        explicit QuasiTimeSynchronization(StreamFileTrain& train);
        QuasiTimeSynchronization(const QuasiTimeSynchronization& other) = delete;
        QuasiTimeSynchronization(QuasiTimeSynchronization&& other) noexcept;

        QuasiTimeSynchronization& operator=(const QuasiTimeSynchronization& other) = delete;
        QuasiTimeSynchronization& operator=(QuasiTimeSynchronization&& other) noexcept;

        [[nodiscard]] const char* getName() const noexcept override { return "quasi-time synchronization"; }

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASITIMESYNCHRONIZATION_H
