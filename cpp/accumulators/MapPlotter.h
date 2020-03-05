//
// Created by serik1987 on 16.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPPLOTTER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPPLOTTER_H

#include "FrameAccumulator.h"


namespace GLOBAL_NAMESPACE {

    /**
     * This accumulator plots single map from the native data
     */
    class MapPlotter: public FrameAccumulator {
    protected:
        enum MapTypes {Real = 0, Imag = 1};

        void initializeBuffers() override;
        void processFrameData(int timestamp) override;

    public:

        explicit MapPlotter(Isoline& isoline);

        MapPlotter(const MapPlotter& other);

        MapPlotter(MapPlotter&& other) noexcept;

        MapPlotter& operator=(const MapPlotter& other);

        MapPlotter& operator=(MapPlotter&& other) noexcept;

        ~MapPlotter() override;

        void clearState() override;

        /**
         *
         * @return the real part of the map
         */
        [[nodiscard]] const double* getRealMap() const {
            if (accumulated && resultMapList.size() >= 2){
                return resultMapList[0];
            } else {
                throw NotAccumulatedException();
            }
        }

        [[nodiscard]] const double* getImaginaryMap() const {
            if (accumulated && resultMapList.size() >= 2){
                return resultMapList[1];
            } else {
                throw NotAccumulatedException();
            }
        }

        [[nodiscard]] std::string getName() const override { return "MAP PLOTTER"; }


    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPPLOTTER_H
