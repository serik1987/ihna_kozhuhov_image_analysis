//
// Created by serik1987 on 14.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ACCUMULATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ACCUMULATOR_H

#include "../isolines/Isoline.h"


namespace GLOBAL_NAMESPACE {

    /**
     * A base class for all "accumulators"
     *
     * Accumulator is an object that gathers all frames and plots the data
     * averaged for all frames, except TraceReader and TraceReaderAndCleaner
     * that represent the data for individual frames
     */
    class Accumulator {
    private:
        double* readingBuffer;
        Isoline* isoline;

    protected:
        bool accumulated;

        /**
         * Prints specilities suitable for the special class only
         */
        virtual void printSpecial(std::ostream& out) const = 0;

        ProgressFunction progressFunction;
        void* progressHandle;

        /**
         * Initializes the accumulation process, including:
         *
         * - doing synchronization
         * - creating all necessary buffer
         * - initialization of the isoline remover
         */
        void initialize();

        /**
         * Initializes memory for all necessary buffers
         */
        virtual void initializeBuffers();

    public:

        /**
         * Creates new Accumulator object
         *
         * @param isoline reference to the Isoline object that will be used for the isoline remove
         */
        explicit Accumulator(Isoline& isoline);

        /**
         * Creates an accumulator that is an exact copy of the other accumulatore
         *
         * @param other the other accumulator
         */
        Accumulator(const Accumulator& other);

        /**
         * Moves the accumulator from the other accumulator. The other accumulator will be corrupted during
         * this action
         *
         * @param other the other accumulator
         */
        Accumulator(Accumulator&& other) noexcept;

        /**
         * Copies the accumulator other to the current accumulator
         *
         * @param other source
         * @return reference to the target
         */
        Accumulator& operator=(const Accumulator& other);

        /**
         * Moves the accumulator other to the current accumulator
         *
         * @param other the source
         * @return reference to the target
         */
        Accumulator& operator=(Accumulator&& other) noexcept;

        /**
         * Destroys the accumulator
         */
        virtual ~Accumulator();

        /**
         * Clears the accumulator state
         */
        virtual void clearState();

        /**
         * Returns the calculator name
         */
        [[nodiscard]] virtual std::string getName() const = 0;

        /**
         *
         * @return total number of channels
         */
        [[nodiscard]] virtual int getChannelNumber() const = 0;

        /**
         *
         * @return reference to the isoline applied
         */
        [[nodiscard]] Isoline& getIsoline() { return *isoline; }

        /**
         *
         * @return reference to the synchronization applied
         */
        Synchronization& getSynchronization() { return isoline->sync(); }

        /**
         *
         * @return reference to the using train
         */
        StreamFileTrain& getTrain() { return getSynchronization().getTrain(); }

        /**
         *
         * @return reference to the train suitable for read-only purpose
         */
        [[nodiscard]] StreamFileTrain& getTrain() const {
            return isoline->sync().getTrain();
        }

        /**
         *
         * @return pointer to the reading buffer (for internal use only)
         */
        double* getReadingBuffer() { return readingBuffer; }

        /**
         * Puts text information about the accumulator into the output stream
         *
         * @param out reference to the output stream
         * @param accumulator reference to the accumulator
         * @return reference to the output stream
         */
        friend std::ostream& operator<<(std::ostream& out, const Accumulator& accumulator);

        /**
         *
         * @return true if the accumulator is accumulated
         */
        [[nodiscard]] bool isAccumulated() const { return accumulated; }

        /**
         * Performs the accumulation process
         */
        void accumulate();

        /**
         * Sets the progress function. This function will be executed after processing each 100 frames
         * and shall be used for update of the progress bar
         *
         * @param f pointer to the progress function
         * @param handle
         */
        void setProgressFunction(ProgressFunction f, void* handle){
            progressFunction = f;
            progressHandle = handle;
        }




        class AccumulatorException: public iman_exception{
        public:
            explicit AccumulatorException(const std::string& msg): iman_exception(msg) {};
        };

        class NotAccumulatedException: public AccumulatorException{
        public:
            NotAccumulatedException(): AccumulatorException("Please, accumulate() the data to apply this function") {};
        };

        class InterruptedException: public AccumulatorException{
        public:
            InterruptedException(): AccumulatorException("The action is interrupted by the user") {};
        };

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ACCUMULATOR_H
