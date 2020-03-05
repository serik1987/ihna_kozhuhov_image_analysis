//
// Created by serik1987 on 15.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAMEACCUMULATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAMEACCUMULATOR_H

#include <vector>
#include "Accumulator.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This is the base class for all accumulators that require the whole frame for the processing
     */
    class FrameAccumulator: public Accumulator {
    private:
        bool preprocessFilter;
        int preprocessFilterRadius;
        bool divideByAverage;

    protected:

        std::vector<double*> resultMapList;
        double* filterBuffer;

        void initializeBuffers() override;

        void printSpecial(std::ostream& out) const override;

    public:
        /**
         * Accumulator initialization
         *
         * @param isoline object that will be used for the isoline removal
         */
        explicit FrameAccumulator(Isoline& isoline);

        FrameAccumulator(const FrameAccumulator& other);

        FrameAccumulator(FrameAccumulator&& other) noexcept;

        FrameAccumulator& operator=(const FrameAccumulator& other);

        FrameAccumulator& operator=(FrameAccumulator&& other) noexcept;

        ~FrameAccumulator() override;

        /**
         * Clears the state and all buffers associated with it
         */
        void clearState() override;

        /**
         * Preprocessing filter is a spatial low-pass filter that will be applied immediately after the frame
         * reading, before the average remove
         *
         * @return true if preprocessing filter is ON, false otherwise
         */
        [[nodiscard]] bool getPreprocessFilter() const { return preprocessFilter; }

        /**
         * Preprocessing filter is a spatial low-pass filter that will be applied immediately after the frame
         * reading, before the average remove. This function activates or deactivates the preprocessing filter
         *
         * @param value true to activate the filter, false to deactivate this. After activation the preprocessing
         * filter radius will be set to 3 px. To change the radius use setPreprocessFilterRadius(...) function
         */
        void setPreprocessFilter(bool value) {
            preprocessFilter = value;
            if (value){
                preprocessFilterRadius = 3;
            }
        }

        /**
         * Preprocessing filter is a spatial low-pass filter that will be applied immediately after the frame
         * reading, before the average remove. The filter radius defines the low-pass band size which is the highest
         * if the radius is 1 and the lowest if the radius is maximum
         *
         * @return filter radius in px
         */
        [[nodiscard]] int getPreprocessFilterRadius() const { return preprocessFilterRadius; }

        /**
         * Preprocessing filter is a spatial low-pass filter that will be applied immediately after the frame
         * reading, before the average remove. The filter radius defines the low-pass band size which is the highest
         * if the radius is 1 and the lowest if the radius is maximum
         *
         * @param r value of the preprocess filter radius
         */
        void setPreprocessFilterRadius(int r){
            int x_size = getTrain().getXSize();
            int y_size = getTrain().getYSize();
            int min_size = x_size < y_size ? x_size : y_size;

            if (r > 0 && r <= min_size){
                preprocessFilterRadius = r;
            } else {
                throw BadPreprocessFilterRadiusException();
            }
        }

        /**
         * If divide by average option is ON the object will average all frames. The average will be done after
         * the preprocess filtration but before the isoline remove. After the final results are gained, the very
         * last stage is division of these final result by this averaged frame
         *
         * @return true if this option is ON, false otherwise
         */
        [[nodiscard]] bool isDivideByAverage() const { return divideByAverage; }

        /**
         * If divide by average option is ON the object will average all frames. The average will be done after
         * the preprocess filtration but before the isoline remove. After the final results are gained, the very
         * last stage is division of these final result by this averaged frame
         *
         * @return true if this option is ON, false otherwise
         */
        void setDivideByAverage(bool value) { divideByAverage = value; }

        /**
         *
         * @return total number of channels
         */
        [[nodiscard]] int getChannelNumber() const override { return getTrain().getXYSize(); }

        /**
         * Reads the whole frame data from the buffer.
         * Additionally, provides preprocess spatial LPF
         *
         * @param frameNumber number of frames to read
         * @return pointer to the reading buffer
         */
        double* readFrameData(int frameNumber) override;


        class BadPreprocessFilterRadiusException: public AccumulatorException{
        public:
            BadPreprocessFilterRadiusException():
                AccumulatorException("Bad value of the preprocess filter radius") {};
        };


    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAMEACCUMULATOR_H
