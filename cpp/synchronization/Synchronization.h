//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SYNCHRONIZATION_H

#include "../exceptions.h"
#include "../source_files/StreamFileTrain.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This is the base class to provide synchronization.
     *
     * The synchronization process contains three steps:
     * 1) During the continuous stimulation it reads the reference signal. Such a signal
     * reflects change in dependency of stimulus phase on the timestamp
     * 2) Both during the continuous and the episodic stimulation the synchronization
     * defines the initial frame and the final frame (i.e., borders of stimulation start and
     * finish)
     * 3) In the continuous stimulation, the synchronization defines the reference cosine and
     * sine. In order to obtain functional maps the recorded signal shall be scalar-producted
     * by the reference sine and cosine.
     */
    class Synchronization {
    private:
        bool doPrecise;
        bool synchronized;
        double* referenceSignalCos;
        double* referenceSignalSin;
        double harmonic;

    protected:
        int initialFrame;
        int finalFrame;
        StreamFileTrain& train;
        double* synchronizationPhase;
        double phaseIncrement;
        double initialPhase;

    public:
        /**
         * Creating constructor
         *
         * @param train reference to the stream file train. The train is assumed to be open()'ed
         */
        explicit Synchronization(StreamFileTrain& train);
        Synchronization(const Synchronization& other) = delete;
        Synchronization(Synchronization&& other) noexcept;
        virtual ~Synchronization();

        Synchronization& operator=(const Synchronization& other) = delete;
        Synchronization& operator=(Synchronization&& other) noexcept;

        /**
         *
         * @return pointer to a short C string containing description of a certain type of synchronization
         */
        [[nodiscard]] virtual const char* getName() const noexcept = 0;

        /**
         *
         * @return the initial frame of the selected record
         */
        [[nodiscard]] int getInitialFrame() const { return initialFrame; }

        /**
         *
         * @return the final frame of the selected record
         */
        [[nodiscard]] int getFinalFrame() const { return finalFrame; }

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SYNCHRONIZATION_H
