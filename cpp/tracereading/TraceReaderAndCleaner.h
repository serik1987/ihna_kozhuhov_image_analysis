//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADERANDCLEANER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADERANDCLEANER_H

#include "TraceReader.h"
#include "../isolines/Isoline.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This is a derived class from the TraceReader. Instead of TraceReader this also cleans
     * the trace by application of a certain isoline which shall be set by the setIsoline method
     */
    class TraceReaderAndCleaner: public TraceReader {
    private:
        double* tracesBeforeRemove;
        double* isolines;
        Isoline* isolineRemover;
        int offsetFrame;

        bool cleaned;

        void newBuffers();

    public:
        explicit TraceReaderAndCleaner(StreamFileTrain& train);
        TraceReaderAndCleaner(const TraceReaderAndCleaner& other) = delete;
        TraceReaderAndCleaner(TraceReaderAndCleaner&& other) noexcept;
        ~TraceReaderAndCleaner() override;

        TraceReaderAndCleaner& operator=(const TraceReaderAndCleaner& other) = delete;
        TraceReaderAndCleaner& operator=(TraceReaderAndCleaner&& other) noexcept;

        /**
         * Sets the Isoline object that will remove isolines from given traces
         *
         * @param isoline reference to the object
         */
        void setIsolineRemover(Isoline& isoline) { isolineRemover = &isoline; }

        /**
         *
         * @return true if all isolines were successfully read and cleaned
         */
        [[nodiscard]] bool isCleaned() const { return hasRead() && cleaned; }

        /**
         *
         * @return a 2D C-type array containing all isolines for all data pixels and native signals
         * for data pixels
         */
        [[nodiscard]] const double* getIsolines() const;

        /**
         * Reads the traces and cleans them from the trash;
         */
        void read() override;

        /**
         * Clears the state from the previous reading
         */
        void clearState() override;

        /**
         *
         * @return traces before remove
         */
        [[nodiscard]] const double* getTracesBeforeRemove() const;

        [[nodiscard]] const char* getReaderName() const override { return "TRACE READER AND CLEANER"; };

        friend std::ostream& operator<<(std::ostream& out, TraceReaderAndCleaner& reader);
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADERANDCLEANER_H
