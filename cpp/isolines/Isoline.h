//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H

#include <functional>
#include "../synchronization/Synchronization.h"

namespace GLOBAL_NAMESPACE {

    class TraceReaderAndCleaner;
    class Accumulator;

    /**
     * This is the base class for all objects that provide the isoline remove. This class is abstract, you can't
     * use it in anyway. However, you may use some of its derived classes each of which corresponds to a certain
     * algorithm to the isoline remove
     *
     * Use TraceReaderAndCleaner to apply this object for isoline remove from traces
     */
    class Isoline {
    private:
        Synchronization* psync;
        StreamFileTrain* ptrain;

    protected:
        int offset;

        int analysisInitialCycle;
        int analysisFinalCycle;
        int isolineInitialCycle;
        int isolineFinalCycle;

        int analysisInitialFrame;
        int analysisFinalFrame;
        int isolineInitialFrame;
        int isolineFinalFrame;

        ProgressFunction progressFunction;
        void* progressHandle;

        virtual void printSpecial(std::ostream& out) const = 0;

        bool removed;

    public:
        Isoline(StreamFileTrain& train, Synchronization& sync);
        Isoline(const Isoline& other);
        virtual ~Isoline() = default;

        /**
         *
         * @return true if the isoline is removed
         */
        [[nodiscard]] bool isRemoved() const { return removed; }

        Isoline& operator=(const Isoline& other) noexcept;

        [[nodiscard]] virtual const char* getName() const noexcept = 0;

        Synchronization& sync() { return *psync; }
        StreamFileTrain& train() { return *ptrain; }

        /**
         *
         * @return number of frames at the beginning of the isoline that are excluded from the analysis
         * or -1 if the isoline has not been removed
         */
        [[nodiscard]] int getFrameOffset() const { return offset; }

        /**
         *
         * @return initial cycle for the analysis or -1 if the isoline has not been removed
         */
        [[nodiscard]] int getAnalysisInitialCycle() const { return analysisInitialCycle; }

        /**
         *
         * @return final cycle for the analysis or -1 if the isoline has not been removed
         */
        [[nodiscard]] int getAnalysisFinalCycle() const { return analysisFinalCycle; }

        /**
         *
         * @return initial cycle, used for isoline calculation
         */
        [[nodiscard]] int getIsolineInitialCycle() const { return isolineInitialCycle; }

        /**
         *
         * @return final cycle used for isoline calculation
         */
        [[nodiscard]] int getIsolineFinalCycle() const { return isolineFinalCycle; }

        /**
         *
         * @return the first frame included into the analysis
         */
        [[nodiscard]] int getAnalysisInitialFrame() const { return analysisInitialFrame; }

        /**
         *
         * @return the last frame included into the analysis
         */
        [[nodiscard]] int getAnalysisFinalFrame() const { return analysisFinalFrame; }

        /**
         *
         * @return the first frame participated in the isoline calculation
         */
        [[nodiscard]] int getIsolineInitialFrame() const { return isolineInitialFrame; }

        /**
         *
         * @return the last frame participated in the isoline calculation
         */
        [[nodiscard]] int getIsolineFinalFrame() const { return isolineFinalFrame; }

        friend std::ostream& operator<<(std::ostream& out, const Isoline& isoline);

        /**
         * Clears the state
         */
        virtual void clearState();

        /**
         * Extends the isoline range. Please, apply this method immediately before the synchronization of isolines
         * This method shall call sync()->setInitialCycle() and sync()->setFinalCycle in the long run
         */
        virtual void extendRange() = 0;

        /**
         * Synchronizes the isolines. Please, apply this method before reading the dirty data if reading the dirty
         * data is necessary
         */
        void synchronizeIsolines();

        /**
         * This method shall be run after the isolines were synchronized in order to adjust the Synchronization from
         * the isolines range into the analysis range. This method shall call sync()->setInitialCycle() and
         * sync()->setFinalCycle() in the long run
         */
        virtual void sacrifice() = 0;

        /**
         * Synchronizes the analysis epoch. This method shall be called immediately after sacrifice() method and
         * before the isoline cleaning itself
         */
        void synchronizeSignal();

        /**
         * Initializes the isoline, i.e., prepares it for the following usage of some Accumulator.
         * After the initialization the isoline shall be used only by a certain accumulator. Its usage by
         * any other accumulators may result to fail
         */
        virtual void initialize(Accumulator& accumulator) = 0;

        void setProgressFunction(ProgressFunction function, void* handle){
            progressFunction = function;
            progressHandle = handle;
        }

        /**
         * Computes the isolines from stand-alone traces that have already been read
         *
         * @param reader reference to the TraceReaderAndCleaner
         * @param src pointer to the array that contains traces without isoline remove, src corresponds to timestamp 0
         * @param srcFirst left border of this array. Don't go beyond this point!
         * @param srcLast right point of this array. Don't go beyond this point!
         * @param isolines output array where all isolines shall be written
         */
        virtual void traceCleaning(TraceReaderAndCleaner& reader, const double* src, const double* srcFirst,
                const double* srcLast, double* isolines) = 0;

        /**
         *
         * @return number of frames that shall be used for isoline plotting
         */
        [[nodiscard]] int getIsolineFrameNumber() const { return isolineFinalFrame - isolineInitialFrame + 1; }

        /**
         *
         * @return numnber of frames for analysis
         */
        [[nodiscard]] int getAnalysisFrameNumber() const { return analysisFinalFrame - analysisInitialFrame + 1; }


        class IsolineException: public iman_exception{
        public:
            explicit IsolineException(const std::string& msg): iman_exception(msg) {};
        };
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
