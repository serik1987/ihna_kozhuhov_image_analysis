//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H

#include <functional>
#include "../synchronization/Synchronization.h"

namespace GLOBAL_NAMESPACE {

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

        Synchronization& sync() { return *psync; }
        StreamFileTrain& train() { return *ptrain; }

        ProgressFunction progressFunction;

        virtual void printSpecial(std::ostream& out) const = 0;

    public:
        Isoline(StreamFileTrain& train, Synchronization& sync);
        Isoline(const Isoline& other);
        virtual ~Isoline() = default;

        Isoline& operator=(const Isoline& other) noexcept;

        [[nodiscard]] virtual const char* getName() const noexcept = 0;

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


        class IsolineException: public iman_exception{
        public:
            explicit IsolineException(const std::string& msg): iman_exception(msg) {};
        };
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
