//
// Created by serik1987 on 15.02.2020.
//

#include "TraceAutoReader.h"

namespace GLOBAL_NAMESPACE {

    TraceAutoReader::TraceAutoReader(Isoline &isoline) :
        Accumulator(isoline), TraceReader(isoline.sync().getTrain()) {

        time = -1.0;
        times = nullptr;
        averagedSignal = nullptr;

    }

    TraceAutoReader::TraceAutoReader(TraceAutoReader &&other) noexcept: Accumulator(std::move(other)),
        TraceReader(std::move(other)){

        time = other.time;
        times = other.times;
        other.times = nullptr;
        averagedSignal = other.averagedSignal;
        other.averagedSignal = nullptr;

    }

    TraceAutoReader &TraceAutoReader::operator=(TraceAutoReader &&other) {
        Accumulator::operator=(std::move(other));
        TraceReader::operator=(std::move(other));

        time = other.time;
        times = other.times;
        other.times = nullptr;
        averagedSignal = other.averagedSignal;
        other.averagedSignal = nullptr;

        return *this;
    }

    TraceAutoReader::~TraceAutoReader() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE TRACE AUTO READER\n";
#endif
        delete [] times;
        delete [] averagedSignal;
    }

    void TraceAutoReader::clearState() {
        Accumulator::clearState();
        delete [] times;
        times = nullptr;
        delete [] averagedSignal;
        averagedSignal = nullptr;
    }

    const double *TraceAutoReader::getTimes() const {
        if (accumulated && times != nullptr){
            return times;
        } else {
            throw NotAccumulatedException();
        }
    }

    const double *TraceAutoReader::getAveragedSignal() const {
        if (accumulated && averagedSignal != nullptr){
            return averagedSignal;
        } else {
            throw NotAccumulatedException();
        }
    }

    void TraceAutoReader::printSpecial(std::ostream &out) const {
        out << "Frame number: " << getFrameNumber() << std::endl;
    }

    void TraceAutoReader::initializeBuffers() {
        Accumulator::initializeBuffers();
        unsigned int N = isoline->getAnalysisFrameNumber();
        times = new double[N];
        averagedSignal = new double[N];
        for (unsigned int i = 0; i < N; ++i){
            times[i] = 0.0;
            averagedSignal[i] = 0.0;
        }
    }


}