//
// Created by serik1987 on 12.01.2020.
//

#include "TraceReaderAndCleaner.h"

namespace GLOBAL_NAMESPACE {

    TraceReaderAndCleaner::TraceReaderAndCleaner(StreamFileTrain &train) : TraceReader(train) {
        tracesBeforeRemove = nullptr;
        isolines = nullptr;
        isolineRemover = nullptr;
        cleaned = false;
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

    const double *TraceReaderAndCleaner::getTracesBeforeRemove() const {
        if (hasRead() && tracesBeforeRemove != nullptr){
            return tracesBeforeRemove;
        } else {
            throw TracesNotReadException();
        }
    }

    const double *TraceReaderAndCleaner::getIsolines() const {
        if (cleaned && isolines != nullptr){
            return isolines;
        } else {
            throw TracesNotReadException();
        }
    }

    std::ostream &operator<<(std::ostream &out, TraceReaderAndCleaner &reader) {
        TraceReader* handle = &reader;

        out << *handle;
        if (reader.isCleaned()){
            out << "Traces were cleaned\n";
        } else {
            out << "Traces were not cleaned\n";
        }

        return out;
    }
}