//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H

#include <list>
#include "../exceptions.h"
#include "../source_files/StreamFileTrain.h"
#include "PixelListItem.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This class provides the reading of stand-alone traces, synchronization channels, arrival times
     */
    class TraceReader {
    public:
        typedef int (*ProgressFunction)(int steps_completed, int steps_total);
        enum PointType {Uint16, Uint32, Uint64};

    private:
        StreamFileTrain& train;
        bool hasRead;

        int initialFrame;
        int finalFrame;
        std::list<PixelListItem> pixelList;

        size_t initialDisplacement;
        size_t* dataDisplacements;
        PointType* dataTypes;

        double* traces;

    public:
        explicit TraceReader(StreamFileTrain& train);
        TraceReader(const TraceReader& other) = delete;
        TraceReader(TraceReader&& other) noexcept;
        virtual ~TraceReader();

        TraceReader& operator=(const TraceReader& other) = delete;
        TraceReader& operator=(TraceReader&& other);

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H
