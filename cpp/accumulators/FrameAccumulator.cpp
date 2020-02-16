//
// Created by serik1987 on 15.02.2020.
//

#include "FrameAccumulator.h"

namespace GLOBAL_NAMESPACE {

    FrameAccumulator::FrameAccumulator(Isoline &isoline) : Accumulator(isoline), resultMapList() {
        preprocessFilter = false;
        preprocessFilterRadius = -1;
        divideByAverage = false;
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
        for (double* buffer: other.resultMapList){
            auto* new_buffer = new double[other.getChannelNumber()];
            std::memcpy(new_buffer, buffer, other.getChannelNumber() * sizeof(double));
            resultMapList.push_back(new_buffer);
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

        return *this;
    }

    FrameAccumulator::~FrameAccumulator() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE FRAME ACCUMULATOR\n";
#endif
        for (double* buffer: resultMapList){
            delete [] buffer;
        }
    }

    void FrameAccumulator::clearState() {
        Accumulator::clearState();

        for (double* buffer: resultMapList){
            delete [] buffer;
        }
        resultMapList.clear();
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
}