//
// Created by serik1987 on 15.02.2020.
//

#include <algorithm>
#include "TraceAutoReader.h"
#include "../source_files/Frame.h"

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
        std::sort(pixelList.begin(), pixelList.end());
        auto final_iterator = std::unique(pixelList.begin(), pixelList.end());
        pixelList.erase(final_iterator, pixelList.end());
        signalNorm = 0.0;
    }

    double *TraceAutoReader::readFrameData(int frameNumber) {
        auto* readingBuffer = getReadingBuffer();

        auto& train = isoline->sync().getTrain();
        auto& frame = train[frameNumber];
        time = frame.getFramChunk().getTimeArrival();

        auto* readingIt = readingBuffer;
        auto* frameBody = frame.getBody();
        unsigned int X = train.getXSize();
        for (int i = 0; i < getChannelNumber(); ++i){
            auto item = getPixelItem(i);
            if (item.getPointType() == PixelListItem::PixelValue){
                int y = item.getI();
                int x = item.getJ();
                unsigned int P = y * X + x;
                *(readingIt++) = frameBody[P];
            } else {
                std::cout << "WARNING. CHANNEL WILL BE IGNORED\n";
            }
        }

        return readingBuffer;
    }

    void TraceAutoReader::processFrameData(int timestamp) {
        times[timestamp] = time;

        auto* readingBuffer = getReadingBuffer();
        double S = 0.0;
        for (int i=0; i < getChannelNumber(); ++i){
            S += readingBuffer[i];
        }
        averagedSignal[timestamp] = S / getChannelNumber();
    }

    void TraceAutoReader::framePreprocessing(int frameNumber, int timestamp) {
        double* readingBuffer = getReadingBuffer();
        double avgValue = 0.0;
        for (int i=0; i < getChannelNumber(); ++i){
            avgValue += readingBuffer[i];
        }
        avgValue /= getChannelNumber();
        signalNorm += avgValue;
    }

    void TraceAutoReader::finalize() {
        unsigned int N = isoline->getAnalysisFrameNumber();
        signalNorm /= N;
        double factor = 100.0 / signalNorm;
        for (unsigned int i=0; i < N; ++i){
            averagedSignal[i] *= factor;
        }
    }


}