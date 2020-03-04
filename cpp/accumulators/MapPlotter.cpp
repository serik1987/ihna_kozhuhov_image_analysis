//
// Created by serik1987 on 16.02.2020.
//

#include "MapPlotter.h"

namespace GLOBAL_NAMESPACE {

    MapPlotter::MapPlotter(Isoline &isoline) : FrameAccumulator(isoline) {

    }

    MapPlotter::MapPlotter(const MapPlotter &other): FrameAccumulator(other) {

    }

    MapPlotter::MapPlotter(MapPlotter &&other) noexcept: FrameAccumulator(std::move(other)) {

    }

    MapPlotter &MapPlotter::operator=(const MapPlotter &other) {
        if (&other == this){
            return *this;
        }
        FrameAccumulator::operator=(other);

        return *this;
    }

    MapPlotter &MapPlotter::operator=(MapPlotter &&other) noexcept {
        if (&other == this){
            return *this;
        }
        FrameAccumulator::operator=(std::move(other));

        return *this;
    }

    MapPlotter::~MapPlotter() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "MAP PLOTTER DESTRUCTION\n";
#endif
    }

    void MapPlotter::clearState() {
        FrameAccumulator::clearState();
    }

    void MapPlotter::initializeBuffers() {
        Accumulator::initializeBuffers();
        resultMapList.push_back(new double[getChannelNumber()]);
        resultMapList.push_back(new double[getChannelNumber()]);
        for (int i=0; i < getChannelNumber(); ++i){
            resultMapList[0][i] = 0.0;
            resultMapList[1][i] = 0.0;
        }
    }
}