//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H

#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This synchronization is based on so called "synchronization channel" that contains signal received from the
     * stimulus synchronizer.
     */
    class ExternalSynchronization: public Synchronization {
    private:
        int synchronizationChannel;
        double* synchronizationSignal;

        int initialCycle;
        int finalCycle;

    public:
        explicit ExternalSynchronization(StreamFileTrain& train);
        ExternalSynchronization(const ExternalSynchronization& other) = delete;
        ExternalSynchronization(ExternalSynchronization&& other) noexcept;
        ~ExternalSynchronization() override;

        ExternalSynchronization& operator=(const ExternalSynchronization& other) = delete;
        ExternalSynchronization& operator=(ExternalSynchronization&& other) noexcept;

        [[nodiscard]] const char* getName() const noexcept override { return "external synchronization"; }

        /**
         *
         * @return number of the synchronization channel containing the recorded reference signal
         */
        [[nodiscard]] int getSynchronizationChannel() const { return synchronizationChannel; }

        /**
         *
         * @return number of the initial cycle.
         * In case where the initial cycle has not been set, the function will return -1 before the synchronization
         * This means that the initial cycle will be set automatically during the synchronization in such a way
         * as to maximize the analysis epoch
         */
        [[nodiscard]] int getInitialCycle() const { return initialCycle; }

        /**
         *
         * @return number of the final cycle
         * In case where the final cycle has not been setm the function will return -1 before the synchronization.
         * This means that the final cycle will be set automatically during the synchronization in such a way as
         * to maximize the analysis epoch
         */
        [[nodiscard]] int getFinalCycle() const { return finalCycle; }

        /**
         * Sets the synchronization channel
         *
         * @param chan index of the synchronization channel
         */
        void setSynchronizationChannel(int chan);

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


        class SynchronizationChannelException: public SynchronizationException{
        public:
            SynchronizationChannelException():
                SynchronizationException("Bad number of the synchronization channel") {};
        };

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H
