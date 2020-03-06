//
// Created by serik1987 on 15.02.2020.
//

#include "FrameAccumulator.h"
#include "../source_files/Frame.h"

namespace GLOBAL_NAMESPACE {

    FrameAccumulator::FrameAccumulator(Isoline &isoline) : Accumulator(isoline), resultMapList() {
        preprocessFilter = false;
        preprocessFilterRadius = -1;
        divideByAverage = false;
        filterBuffer = nullptr;
    }

    FrameAccumulator::FrameAccumulator(const FrameAccumulator &other): Accumulator(other) {
        preprocessFilter = other.preprocessFilter;
        preprocessFilterRadius = other.preprocessFilterRadius;
        divideByAverage = other.divideByAverage;
        resultMapList.clear();
        for (double* buffer: other.resultMapList){
            auto* new_buffer = new double[other.getChannelNumber()];
            std::memcpy(new_buffer, buffer, other.getChannelNumber() * sizeof(double));
            resultMapList.push_back(new_buffer);
        }
        if (other.filterBuffer == nullptr){
            filterBuffer = nullptr;
        } else {
            filterBuffer = new double[other.getChannelNumber()];
            std::memcpy(filterBuffer, other.filterBuffer, other.getChannelNumber() * sizeof(double));
        }
    }

    FrameAccumulator::FrameAccumulator(FrameAccumulator &&other) noexcept: Accumulator(std::move(other)) {
        preprocessFilter = other.preprocessFilter;
        preprocessFilterRadius = other.preprocessFilterRadius;
        divideByAverage = other.divideByAverage;
        resultMapList.clear();
        for (double* buffer: other.resultMapList){
            resultMapList.push_back(buffer);
        }
        other.resultMapList.clear();
        filterBuffer = other.filterBuffer;
        other.filterBuffer = nullptr;
    }

    FrameAccumulator &FrameAccumulator::operator=(const FrameAccumulator &other) {
        if (&other == this){
            return *this;
        }
        Accumulator::operator=(other);

        preprocessFilter = other.preprocessFilter;
        preprocessFilterRadius = other.preprocessFilterRadius;
        divideByAverage = other.divideByAverage;
        for (double* buffer: resultMapList){
            delete [] buffer;
            buffer = nullptr;
        }
        resultMapList.clear();
        for (double* buffer: other.resultMapList) {
            auto *new_buffer = new double[other.getChannelNumber()];
            std::memcpy(new_buffer, buffer, other.getChannelNumber() * sizeof(double));
            resultMapList.push_back(new_buffer);
        }
        delete [] filterBuffer;
        if (other.filterBuffer == nullptr){
            filterBuffer = nullptr;
        } else {
            filterBuffer = new double[other.getChannelNumber()];
            std::memcpy(filterBuffer, other.filterBuffer, other.getChannelNumber() * sizeof(double));
        }

        return *this;
    }

    FrameAccumulator &FrameAccumulator::operator=(FrameAccumulator &&other) noexcept {
        if (&other == this){
            return *this;
        }
        Accumulator::operator=(std::move(other));

        preprocessFilter = other.preprocessFilter;
        preprocessFilterRadius = other.preprocessFilterRadius;
        divideByAverage = other.divideByAverage;
        for (double* buffer: resultMapList){
            delete [] buffer;
            buffer = nullptr;
        }
        resultMapList.clear();
        for (double* buffer: other.resultMapList){
            resultMapList.push_back(buffer);
        }
        other.resultMapList.clear();

        delete [] filterBuffer;
        filterBuffer = other.filterBuffer;
        other.filterBuffer = nullptr;

        return *this;
    }

    FrameAccumulator::~FrameAccumulator() {
#ifdef DEBUG_DELETE_CHECK
        // std::cout << "DELETE FRAME ACCUMULATOR\n";
#endif
        for (double* buffer: resultMapList){
            delete [] buffer;
        }
        delete [] filterBuffer;
    }

    void FrameAccumulator::clearState() {
        Accumulator::clearState();

        for (double* buffer: resultMapList){
            delete [] buffer;
        }
        resultMapList.clear();

        delete [] filterBuffer;
        filterBuffer = nullptr;
    }

    void FrameAccumulator::printSpecial(std::ostream &out) const {
        if (getPreprocessFilter()){
            out << "Preprocess filter ON\n";
            out << "Preprocess filter radius: " << getPreprocessFilterRadius() << std::endl;
        } else {
            out << "Preprocess filter OFF\n";
        }
        if (isDivideByAverage()){
            out << "Division by average ON\n";
        } else {
            out << "Division by average OFF\n";
        }
        out << "Number of output buffers: " << resultMapList.size() << std::endl;
    }

    double *FrameAccumulator::readFrameData(int frameNumber) {
        auto* readingBuffer = getReadingBuffer();

        StreamFileTrain& train = isoline->sync().getTrain();
        Frame& frame = train[frameNumber];
        auto* body = frame.getBody();

        for (int i = 0; i < train.getXYSize(); ++i){
            readingBuffer[i] = body[i];
        }

        if (preprocessFilter){
            std::memcpy(filterBuffer, readingBuffer, sizeof(double) * getChannelNumber());
            unsigned int Y = train.getYSize();
            unsigned int X = train.getXSize();
            auto* readingIt = readingBuffer;
            for (unsigned int y0 = 0; y0 < Y; ++y0){
                for (unsigned int x0=0; x0 < X; ++x0){
                    unsigned int x_min = x0 - preprocessFilterRadius;
                    if (x_min < 0) x_min = 0;
                    unsigned int y_min = y0 - preprocessFilterRadius;
                    if (y_min < 0) y_min = 0;
                    unsigned int x_max = x0 + preprocessFilterRadius;
                    if (x_max > X-1) x_max = X-1;
                    unsigned int y_max = y0 + preprocessFilterRadius;
                    if (x_max > Y-1) y_max = Y-1;
                    double value = 0.0;
                    int m = 0;
                    for (unsigned int y = y_min; y <= y_max; ++y){
                        for (unsigned int x = x_min; x <= x_max; ++x){
                            unsigned int P = y * X + x;
                            value += filterBuffer[P];
                            m++;
                        }
                    }
                    *(readingIt++) = value / m;
                }
            }
        }

        return readingBuffer;
    }

    void FrameAccumulator::initializeBuffers() {
        Accumulator::initializeBuffers();
        if (preprocessFilter){
            filterBuffer = new double[getChannelNumber()];
        }
    }


}