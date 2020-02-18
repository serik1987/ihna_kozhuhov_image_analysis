//
// Created by serik1987 on 16.02.2020.
//

#include "MapFilter.h"

namespace GLOBAL_NAMESPACE {

    MapFilter::MapFilter(Isoline &isoline) : FrameAccumulator(isoline), b(), a(), sourceMapList() {
        targetMap = nullptr;
    }

    MapFilter::MapFilter(const MapFilter &other): FrameAccumulator(other) {
        for (double* buffer: other.sourceMapList){
            auto* new_buffer = new double[other.getChannelNumber()];
            std::memcpy(new_buffer, buffer, other.getChannelNumber() * sizeof(double));
            sourceMapList.push_back(new_buffer);
        }
        targetMap = new double[other.getChannelNumber()];
        std::memcpy(targetMap, other.targetMap, other.getChannelNumber() * sizeof(double));
        b = other.b;
        a = other.a;
    }

    MapFilter::MapFilter(MapFilter &&other) noexcept: FrameAccumulator(std::move(other)) {
        for (double* buffer: other.sourceMapList){
            sourceMapList.push_back(buffer);
        }
        other.sourceMapList.clear();
        targetMap = other.targetMap;
        other.targetMap = nullptr;
        for (double* buffer: other.sourceMapList){
            sourceMapList.push_back(buffer);
        };
        other.sourceMapList.clear();
        b = std::move(other.b);
        a = std::move(other.a);
    }

    MapFilter &MapFilter::operator=(const MapFilter &other) {
        if (&other == this){
            return *this;
        }
        FrameAccumulator::operator=(other);

        targetMap = new double[other.getChannelNumber()];
        std::memcpy(targetMap, other.targetMap, other.getChannelNumber() * sizeof(double));

        delete [] targetMap;
        for (double* buffer: sourceMapList){
            delete [] buffer;
        }
        sourceMapList.clear();

        for (double* buffer: other.sourceMapList){
            auto* new_buffer = new double[other.getChannelNumber()];
            std::memcpy(new_buffer, buffer, getChannelNumber() * sizeof(double));
            sourceMapList.push_back(new_buffer);
        }
        targetMap = new double[other.getChannelNumber()];
        std::memcpy(targetMap, other.targetMap, other.getChannelNumber() * sizeof(double));

        b = other.b;
        a = other.a;

        return *this;
    }

    MapFilter &MapFilter::operator=(MapFilter &&other) noexcept {
        if (&other == this){
            return *this;
        }

        delete [] targetMap;
        for (double* buffer: sourceMapList){
            delete [] buffer;
        }
        sourceMapList.clear();

        for (double* buffer: other.sourceMapList){
            sourceMapList.push_back(buffer);
        }
        other.sourceMapList.clear();

        targetMap = other.targetMap;
        other.targetMap = nullptr;

        b = std::move(other.b);
        a = std::move(other.a);

        return *this;
    }

    MapFilter::~MapFilter() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "Map filter destruction\n";
#endif
        delete [] targetMap;
        for (double* buffer: sourceMapList){
            delete [] buffer;
        }
    }

    void MapFilter::clearState() {
        FrameAccumulator::clearState();
        delete [] targetMap;
        targetMap = nullptr;
        for (double* buffer: sourceMapList){
            delete [] buffer;
        }
        sourceMapList.clear();
        b.clear();
        a.clear();
    }

    void MapFilter::printSpecial(std::ostream &out) const {
        FrameAccumulator::printSpecial(out);

        out << "Filter coefficients b (nominator): ";
        for (const double& coefficient: b){
            out << coefficient << " ";
        }
        out << "\nFilter coefficients a (denominator): ";
        for (const double& coefficients: a){
            out << coefficients << " ";
        }
        out << "\n";
    }

}