//
// Created by serik1987 on 16.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPFILTER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPFILTER_H

#include "FrameAccumulator.h"

namespace GLOBAL_NAMESPACE {

    /**
     * Allows to build the distribution of the signal power across the given frequency range
     */
    class MapFilter: public FrameAccumulator {
    private:
        std::vector<double> b;
        std::vector<double> a;

        std::vector<double*> sourceMapList;
        double* targetMap;

    protected:
        void printSpecial(std::ostream& out) const override;

        void initializeBuffers() override;
        void processFrameData(int timestamp) override;

        int getFinalizationMapNumber() override { return 1; };
        double* getFinalizationMap(int number) override { return targetMap; };

    public:
        explicit MapFilter(Isoline& isoline);

        MapFilter(const MapFilter& other);

        MapFilter(MapFilter&& other) noexcept;

        MapFilter& operator=(const MapFilter& other);

        MapFilter& operator=(MapFilter&& other) noexcept;

        ~MapFilter() override;

        void clearState() override;

        /**
         * Sets the nominator (b) coefficients
         *
         * @param value the nominator (b) coefficients
         */
        void setB(std::vector<double>& value){
            b = value;
        }

        /**
         * Sets the denominator (a) coefficients
         *
         * @param value the denominator (a) coefficients
         */
        void setA(std::vector<double>& value){
            a = value;
        }

        /**
         *
         * @return target map
         */
        [[nodiscard]] const double* getTargetMap() const;

        /**
         *
         * @return the accumulator name
         */
        [[nodiscard]] std::string getName() const override { return "MAP FILTER"; }

        class FilterNotSetException: AccumulatorException{
        public:
            FilterNotSetException(): AccumulatorException("Please, set the filter coefficients") {};
        };

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPFILTER_H
