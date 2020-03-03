//
// Created by serik1987 on 14.02.2020.
//

#include <cstring>
#include "Accumulator.h"

namespace GLOBAL_NAMESPACE {

    Accumulator::Accumulator(Isoline &isoline) {
        this->isoline = &isoline;
        readingBuffer = nullptr;
        accumulated = false;
        progressFunction = nullptr;
        progressHandle = nullptr;
    }

    Accumulator::Accumulator(const Accumulator &other) {
        isoline = other.isoline;
        accumulated = other.accumulated;
        if (other.readingBuffer == nullptr){
            readingBuffer = nullptr;
        } else {
            readingBuffer = new double[other.getChannelNumber()];
            std::memcpy(readingBuffer, other.readingBuffer, other.getChannelNumber() * sizeof(double));
        }
        progressFunction = other.progressFunction;
        progressHandle = other.progressHandle;
    }

    Accumulator::Accumulator(Accumulator&& other) noexcept{
        isoline = other.isoline;
        accumulated = other.accumulated;
        readingBuffer = other.readingBuffer;
        other.readingBuffer = nullptr;
        progressFunction = other.progressFunction;
        progressHandle = other.progressHandle;
    }

    Accumulator &Accumulator::operator=(const Accumulator &other) {
        if (&other == this){
            return *this;
        }

        isoline = other.isoline;
        accumulated = other.accumulated;
        delete [] readingBuffer;
        readingBuffer = nullptr;
        if (other.readingBuffer == nullptr){
            readingBuffer = nullptr;
        } else {
            readingBuffer = new double[other.getChannelNumber()];
            std::memcpy(readingBuffer, other.readingBuffer, other.getChannelNumber()*sizeof(double));
        }
        progressFunction = other.progressFunction;
        progressHandle = other.progressHandle;

        return *this;
    }

    Accumulator &Accumulator::operator=(Accumulator &&other) noexcept {
        isoline = other.isoline;
        accumulated = other.accumulated;
        readingBuffer = other.readingBuffer;
        progressFunction = other.progressFunction;
        progressHandle = other.progressHandle;

        return *this;
    }

    Accumulator::~Accumulator() {
#ifdef DEBUG_DELETE_CHECK
        std::cout <<"DELETE ACCUMULATOR\n";
#endif
        isoline->clearState();
        delete [] readingBuffer;
    }

    void Accumulator::clearState() {
        accumulated = false;
        isoline->clearState();
        delete[] readingBuffer;
        readingBuffer = nullptr;
    }

    std::ostream &operator<<(std::ostream &out, const Accumulator &accumulator) {
        out << "===== " << accumulator.getName() << " (accumulator) =====\n";
        out << "Channel number: " << accumulator.getChannelNumber() << "\n";
        if (accumulator.isAccumulated()){
            out << "Accumulation completed\n";
        } else {
            out << "Accumulation is not completed\n";
        }
        accumulator.printSpecial(out);

        return out;
    }

    void Accumulator::accumulate() {
        clearState();
        initialize();
        accumulated = true;
    }

    void Accumulator::initialize() {
        isoline->extendRange();
        if (progressFunction != nullptr){
            progressFunction(0, 1, "Synchronization", progressHandle);
            isoline->sync().setProgressFunction(progressFunction, progressHandle);
        }
        isoline->synchronizeIsolines();
        if (!isoline->sync().isSynchronized()){
            clearState();
            throw InterruptedException();
        }
        isoline->sync().clearState();
        isoline->sacrifice();
        isoline->synchronizeSignal();
        if (!isoline->sync().isSynchronized()){
            clearState();
            throw InterruptedException();
        }
        initializeBuffers();
    }

    void Accumulator::initializeBuffers() {
        readingBuffer = new double[getChannelNumber()];
    }
}