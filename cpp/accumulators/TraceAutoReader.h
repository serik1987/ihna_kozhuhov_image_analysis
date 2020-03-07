//
// Created by serik1987 on 15.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEAUTOREADER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEAUTOREADER_H

#include "Accumulator.h"
#include "../tracereading/TraceReader.h"

namespace GLOBAL_NAMESPACE {

    /**
     * A derivative class for the trace reader.
     *
     * The trace reader reads the data from a particular ROI, clears isoline from them and then average them
     * across ROI. Despite TraceReader and TraceReaderAndCleaner, the class works with large ROIs without requiring
     * enormously large amount of memory but doesn't allow to receive individual traces
     */
    class TraceAutoReader: public Accumulator, public TraceReader {
    private:
        double time;
        double* times;
        double* averagedSignal;
        double signalNorm;

    protected:
        void printSpecial(std::ostream& out) const override;

        void initializeBuffers() override;
        void processFrameData(int timestamp) override;
        void framePreprocessing(int frameNumber, int timestamp) override;
        void finalize() override;

    public:
        /**
         * Initializes the object
         *
         * @param isoline reference to the isoline remover
         */
        explicit TraceAutoReader(Isoline& isoline);

        TraceAutoReader(const TraceAutoReader& other) = delete;

        TraceAutoReader(TraceAutoReader&& other) noexcept;

        TraceAutoReader& operator=(const TraceAutoReader& other) = delete;

        TraceAutoReader& operator=(TraceAutoReader&& other);

        ~TraceAutoReader() override;

        /**
         * Removes all buffers from the state
         */
        void clearState() override;

        /**
         *
         * @return string containing the accumulator name
         */
        [[nodiscard]] std::string getName() const override { return "TRACE READER"; }

        /**
         *
         * @return total number of channels
         */
        [[nodiscard]] int getChannelNumber() const override {
            return TraceReader::getChannelNumber();
        }

        /**
         *
         * @return the time vector that represents arrival times in ms
         */
        [[nodiscard]] const double* getTimes() const;

        /**
         *
         * @return the averaged imaging signal
         */
        [[nodiscard]] const double* getAveragedSignal() const;

        /**
         * The same as accumulate()
         */
        void read() override {
            accumulate();
        }

        /**
         * Reads all data related to a particular frame.
         * Values from the individual pixels will be placed into readingBuffer
         * as they are, they will not be averaged across the signal
         *
         * @param frameNumber number of frame to read
         * @return pointer to the readingBuffer
         */
        double* readFrameData(int frameNumber) override;

        friend std::ostream& operator<<(std::ostream& out, const TraceAutoReader& reader) {
            const auto* accumulator = (Accumulator*)&reader;
            out << *accumulator;
            return out;
        }
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEAUTOREADER_H
