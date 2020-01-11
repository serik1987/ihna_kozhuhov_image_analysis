//
// Created by serik1987 on 11.01.2020.
//

#include "TraceReader.h"


namespace GLOBAL_NAMESPACE {

    TraceReader::TraceReader(StreamFileTrain &train): train(train), pixelList() {
        initialFrame = -1;
        finalFrame = -1;

        initialDisplacement = 0;
        dataDisplacements = nullptr;
        dataTypes = nullptr;
        hasRead = false;

        traces = nullptr;
    }

    TraceReader::TraceReader(TraceReader &&other) noexcept: train(other.train), pixelList(other.pixelList) {
        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;

        initialDisplacement = other.initialDisplacement;
        dataDisplacements = other.dataDisplacements;
        other.dataDisplacements = nullptr;
        dataTypes = other.dataTypes;
        other.dataTypes = nullptr;
        hasRead = other.hasRead;

        traces = other.traces;
        other.traces = nullptr;
    }

    TraceReader::~TraceReader() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE TRACE READER\n";
#endif
        delete [] dataDisplacements;
        delete [] dataTypes;
        delete [] traces;
    }

    TraceReader &TraceReader::operator=(TraceReader &&other) {
        train = other.train;

        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;

        initialDisplacement = other.initialDisplacement;
        dataDisplacements = other.dataDisplacements;
        other.dataDisplacements = nullptr;
        dataTypes = other.dataTypes;
        other.dataTypes = nullptr;
        hasRead = other.hasRead;

        traces = other.traces;
        other.traces = nullptr;

        return *this;
    }


}