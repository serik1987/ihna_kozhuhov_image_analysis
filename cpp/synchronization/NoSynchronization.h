//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOSYNCHRONIZATION_H

#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    /**
     * In this type of synchronization the continuous signal is considered to be absent. This means that:
     * 1) The reference signal is a 2PI multiplied by the timestamp number
     * 2) Initial frame and final frame is defined by the user.
     * 3) Reference cosine and reference sine are usually meaningless
     */
    class NoSynchronization: public Synchronization {
    protected:
        void specialPrint(std::ostream& out) const override;

        void calculateSynchronizationPhase() override;
        void calculatePhaseIncrement() override;

    public:
        explicit NoSynchronization(StreamFileTrain& train);
        NoSynchronization(const NoSynchronization& other) = delete;
        NoSynchronization(NoSynchronization&& other) noexcept: Synchronization(std::move(other)) {};

        NoSynchronization& operator=(const NoSynchronization& other) = delete;
        NoSynchronization& operator=(NoSynchronization&& other) noexcept;

        /**
         * Sets the initial frame
         *
         * @param frame
         */
        void setInitialFrame(int frame);

        /**
         * Sets the final frame
         * @param frame
         */
        void setFinalFrame(int frame);

        /**
         *
         * @return the initial frame that is treated as an initial cycle
         */
        [[nodiscard]] int getInitialCycle() const override { return initialFrame+1; }

        /**
         * Sets the virtual final cycle
         *
         * @param n number of the virtual final cycle
         */
        void setInitialCycle(int n) override {
            setInitialFrame(n-1);
        }

        /**
         *
         * @return value of the virtual final cycle
         */
        [[nodiscard]] int getFinalCycle() const override { return finalFrame+1; }

        /**
         * Sets the virtual final cycle
         *
         * @param n value of the virtual final cycle
         */
        void setFinalCycle(int n) override { return setFinalFrame(n-1); }

        /**
         *
         * @return total number of virtual cycles
         */
        [[nodiscard]] int getCycleNumber() const override { return getFrameNumber(); }

        [[nodiscard]] const char* getName() const noexcept override { return "no synchronization"; }

        class FrameRangeException: public SynchronizationException{
        public:
            FrameRangeException(): SynchronizationException("Incorrect analysis range") {};
        };
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOSYNCHRONIZATION_H
