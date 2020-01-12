//
// Created by serik1987 on 12.01.2020.
//

#include "TraceReaderAndCleaner.h"

namespace GLOBAL_NAMESPACE {

    TraceReaderAndCleaner::TraceReaderAndCleaner(StreamFileTrain &train) : TraceReader(train) {
        tracesBeforeRemove = nullptr;
        isolines = nullptr;
        isolineRemover = nullptr;
    }

    TraceReaderAndCleaner::TraceReaderAndCleaner(TraceReaderAndCleaner &&other) noexcept:
        TraceReader(std::move(other)){

        tracesBeforeRemove = other.tracesBeforeRemove;
        other.tracesBeforeRemove = nullptr;
        isolines = other.isolines;
        other.isolines = nullptr;
        isolineRemover = other.isolineRemover;

    }

    TraceReaderAndCleaner::~TraceReaderAndCleaner() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "REMOVE TRACE READER AND CLEANER\n";
#endif
        delete [] tracesBeforeRemove;
        delete [] isolines;
    }

    TraceReaderAndCleaner &TraceReaderAndCleaner::operator=(TraceReaderAndCleaner &&other) noexcept {
        TraceReader::operator=(std::move(other));

        tracesBeforeRemove = other.tracesBeforeRemove;
        other.tracesBeforeRemove = nullptr;
        isolines = other.isolines;
        other.isolines = nullptr;
        isolineRemover = other.isolineRemover;

        return *this;
    }
}