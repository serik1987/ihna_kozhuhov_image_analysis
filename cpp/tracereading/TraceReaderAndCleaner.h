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

    public:
        explicit TraceReaderAndCleaner(StreamFileTrain& train);
        TraceReaderAndCleaner(const TraceReaderAndCleaner& other) = delete;
        TraceReaderAndCleaner(TraceReaderAndCleaner&& other) noexcept;
        ~TraceReaderAndCleaner() override;

        TraceReaderAndCleaner& operator=(const TraceReaderAndCleaner& other) = delete;
        TraceReaderAndCleaner& operator=(TraceReaderAndCleaner&& other) noexcept;
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADERANDCLEANER_H
