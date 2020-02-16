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

        void inverse();
        void inversePrecise(double h);

        static constexpr float ITERATIONM = 0.5;
        static constexpr double EPSILON = 1e-18;

    protected:
        int initialFrame;
        int finalFrame;
        StreamFileTrain& train;
        double* synchronizationPhase;
        double phaseIncrement;
        double initialPhase;

        ProgressFunction progressFunction;
        void* handle;

        virtual void specialPrint(std::ostream& out) const = 0;

        /**
         * Initializes and calculates the following protected data:
         * synchronizationPhase - represents values of stimulus phases at different timestamps
         * initialFrame - the very first frame participated in the analysis
         * finalFrame - the very last frame participated in the analysis
         */
        virtual void calculateSynchronizationPhase() = 0;

        /**
         * Calculates the value of the following scalar protected variables:
         * initialPhase, phaseIncrement
         */
        virtual void calculatePhaseIncrement() = 0;

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

        /**
         *
         * @return total number of frames available for the analysis
         */
        [[nodiscard]] int getFrameNumber() const { return finalFrame - initialFrame + 1; }

        /**
         *
         * @return total number of cycles
         */
        [[nodiscard]] virtual int getCycleNumber() const = 0;

        /**
         *
         * @return true if precise analysis if needed
         */
        [[nodiscard]] bool isDoPrecise() const { return doPrecise; }

        /**
         * Adjusts the precise analysis parameters
         *
         * @param value true if precise analysis is required, false otherwise
         */
        void setDoPrecise(bool value) { doPrecise = value; }

        /**
         *
         * @return synchronization phase dependent on timestamp as contiguous C array
         */
        [[nodiscard]] const double* getSynchronizationPhase() const;

        /**
         *
         * @return difference of stimulus phase between two consequtive frames or
         * 0.0 if the train has not been synchronized
         */
        [[nodiscard]] double getPhaseIncrement() const { return phaseIncrement; }

        /**
         *
         * @return stimulus phase at 0th timestamp or 0.0 if the train has not been synchronized
         */
        [[nodiscard]] double getInitialPhase() const { return initialPhase; }

        /**
         *
         * @return the reference cosine as contiguous C array
         */
        [[nodiscard]] const double* getReferenceSignalCos() const;

        /**
         *
         * @return the reference sine as contiguous C array
         */
        [[nodiscard]] const double* getReferenceSignalSin() const;

        /**
         *
         * @return harmonic
         */
        [[nodiscard]] double getHarmonic() const { return harmonic;}

        /**
         *
         * @param value harmonic value
         */
        void setHarmonic(double value) { harmonic = value; }

        /**
         * Sets the progress function. This function will be called every time where each 100 frames are synchronized
         *
         * @param f The function
         * @param h the value to be passed to progress function when this is called. This may be a pointer to a
         * progress bar window
         */
        void setProgressFunction(ProgressFunction f, void* h){
            progressFunction = f;
            handle = h;
        }

        /**
         * Clears the previous synchronized state
         */
        virtual void clearState();

        /**
         * Synchronizes the signal
         */
        void synchronize();

        /**
         *
         * @return the stream file train
         */
        StreamFileTrain& getTrain() { return train; }

        /**
         *
         * @return true if synchronization was successfully completed
         */
        [[nodiscard]] bool isSynchronized() const { return synchronized; }

        friend std::ostream& operator<<(std::ostream& out, const Synchronization& sync);

        /**
         * Returns the initial cycle or its analogue
         */
        [[nodiscard]] virtual int getInitialCycle() const = 0;

        /**
         * Sets the virtual initial cycle
         *
         * @param n number of the initial cycle
         */
        virtual void setInitialCycle(int n) = 0;

        /**
         *
         * @return the virtual final cycle
         */
        [[nodiscard]] virtual int getFinalCycle() const = 0;

        /**
         * Sets the value of the virtual final cycle
         *
         * @param n number of the virtual final cycle
         */
        virtual void setFinalCycle(int n) = 0;



        class SynchronizationException: public iman_exception{
        public:
            explicit SynchronizationException(const std::string& msg): iman_exception(msg) {};
        };

        class FileNotOpenedException: public SynchronizationException{
        public:
            explicit FileNotOpenedException():
                SynchronizationException("Synchronization failed because the file has not been opened") {};
        };

        class NotSynchronizedException: public SynchronizationException{
        public:
            explicit NotSynchronizedException():
                SynchronizationException
                ("The property is not accessible because the train has not been synchronize()'d") {};
        };

        class StimulusPeriodException: public SynchronizationException{
        public:
            explicit StimulusPeriodException():
                SynchronizationException("Bad value of the stimulus period") {};
        };

        class InitialCycleException: public SynchronizationException{
        public:
            explicit InitialCycleException():
                SynchronizationException("Bad value of the initial cycle") {};
        };

        class FinalCycleException: public SynchronizationException{
        public:
            explicit FinalCycleException():
                SynchronizationException("Bad value of the final cycle") {};
        };

        class BadHarmonicException: public SynchronizationException{
        public:
            BadHarmonicException(): SynchronizationException("Bad value of harmonic") {};
        };

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SYNCHRONIZATION_H
