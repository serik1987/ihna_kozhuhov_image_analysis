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
            return tracesBeforeRemove + offsetFrame * getChannelNumber();
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

    void TraceReaderAndCleaner::read() {
        isolineRemover->clearState();
        isolineRemover->extendRange();
        progressFunction(0, 1, "Synchronization", handle);
        isolineRemover->sync().setProgressFunction(progressFunction, handle);
        isolineRemover->synchronizeIsolines();
        setFrameRange(isolineRemover->sync());
        TraceReader::read();
        tracesBeforeRemove = traces;
        traces = nullptr;
        isolineRemover->sacrifice();
        progressFunction(0, 1, "Synchronization", handle);
        isolineRemover->synchronizeSignal();
        offsetFrame = isolineRemover->getAnalysisInitialFrame() - isolineRemover->getIsolineInitialFrame();
        newBuffers();
        const double* srcLeft = tracesBeforeRemove;
        const double* src = getTracesBeforeRemove();
        const double* srcRight = srcLeft + isolineRemover->getIsolineFrameNumber() * getChannelNumber();
        progressFunction(0, 1, "Trace cleaning", handle);
        isolineRemover->traceCleaning(*this, src, srcLeft, srcRight, isolines);
        int points = getFrameNumber() * getChannelNumber();
        std::transform(tracesBeforeRemove, tracesBeforeRemove + points, isolines, traces,
                [](double x, double y) { return x-y; });
        cleaned = true;
    }

    void TraceReaderAndCleaner::clearState() {
        TraceReader::clearState();
        delete [] tracesBeforeRemove;
        delete [] isolines;
        cleaned = false;
    }

    void TraceReaderAndCleaner::newBuffers() {
        setFrameRange(isolineRemover->sync());
        int points = isolineRemover->getAnalysisFrameNumber() * getChannelNumber();
        isolines = new double[points];
        std::fill(isolines, isolines + points, 0.0);
        traces = new double[points];
        std::fill(traces, traces + points, 0.0);
    }
}