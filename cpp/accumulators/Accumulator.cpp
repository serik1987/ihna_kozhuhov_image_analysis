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
        std::cout << "ACCUMULATOR Clearing state...\n";
        clearState();
        std::cout << "ACCUMULATOR State cleared\n";
        initialize();
        std::cout << "ACCUMULATOR Accumulator initialized\n";
        int timestamp, frameNumber;
        int initFrame = isoline->getAnalysisInitialFrame();
        int finalFrame = isoline->getAnalysisFinalFrame();
        int totalFrames = finalFrame - initFrame + 1;
        std::cout << "Initial frame: " << initFrame << std::endl;
        std::cout << "Final frame: " << finalFrame << std::endl;
        std::cout << "Total frames: " << totalFrames << std::endl;
        for (timestamp = 0, frameNumber = initFrame; frameNumber <= finalFrame; ++timestamp, ++frameNumber){
            std:: cout << "ACCUMULATOR Processing timestamp..." << timestamp << " out of " <<
            (finalFrame - initFrame + 1) << std::endl;
            readFrameData(frameNumber);
            framePreprocessing(frameNumber, timestamp);
            isoline->advance(*this, frameNumber);
            processFrameData(timestamp);
            if (progressFunction != nullptr && timestamp % 100 == 0){
                bool progressStatus = progressFunction(timestamp, totalFrames, "Accumulation", progressHandle);
                if (!progressStatus){
                    clearState();
                    throw InterruptedException();
                }
            }
        }
        std::cout << "ACCUMULATOR All timestamps were processed. Finalization..." << std::endl;
        finalize();
        std::cout << "ACCUMULATOR Finalization completed\n";
        accumulated = true;
    }

    void Accumulator::initialize() {
        isoline->extendRange();
        std::cout << "ACCUMULATE INI Isoline range extended\n";
        if (progressFunction != nullptr){
            progressFunction(0, 1, "Synchronization", progressHandle);
            isoline->sync().setProgressFunction(progressFunction, progressHandle);
        }
        isoline->synchronizeIsolines();
        std::cout << "ACCUMULATE INI Isolines were synchronized\n";
        if (!isoline->sync().isSynchronized()){
            clearState();
            throw InterruptedException();
        }
        isoline->sync().clearState();
        std::cout << "ACCUMULATE INI Synchronization state was cleared\n";
        isoline->sacrifice();
        std::cout << "ACCUMULATE INI Analysis range was sacrifices\n";
        isoline->synchronizeSignal();
        std::cout << "ACCUMULATE INI Analysis epoch was synchronized\n";
        if (!isoline->sync().isSynchronized()){
            clearState();
            throw InterruptedException();
        }
        try {
            initializeBuffers();
            std::cout << "ACCUMULATE INI Buffer were synchronized\n";
            isoline->setProgressFunction(progressFunction, progressHandle);
            isoline->initialize(*this);
            std::cout << "ACCUMULATE INI Isolines were sychronized\n";
        } catch (std::exception& e){
            clearState();
            throw;
        }
    }

    void Accumulator::initializeBuffers() {
        readingBuffer = new double[getChannelNumber()];
    }
}