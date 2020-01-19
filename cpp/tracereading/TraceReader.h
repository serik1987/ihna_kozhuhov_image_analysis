//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H

#include <list>
#include "../exceptions.h"
#include "../source_files/StreamFileTrain.h"
#include "../source_files/FramChunk.h"
#include "../source_files/FramCostChunk.h"
#include "PixelListItem.h"

namespace GLOBAL_NAMESPACE {

    class Synchronization;

    /**
     * This class provides the reading of stand-alone traces, synchronization channels, arrival times
     */
    class TraceReader {

    private:
        StreamFileTrain& train;
        bool _hasRead;

        int initialFrame;
        int finalFrame;
        std::vector<PixelListItem> pixelList;

        size_t initialDisplacement;
        size_t* dataDisplacements;
        PixelListItem::PointType* dataTypes;

        static constexpr size_t ARRIVAL_TIME_DISPLACEMENT =
                sizeof(ChunkHeader::DATA_CHUNK) + offsetof(FramChunk::FRAM_CHUNK, TimeArrivalUsecLo);

        static constexpr size_t SYNCHRONIZATION_CHANNEL_DISPLACEMENT =
                2 * sizeof(ChunkHeader::DATA_CHUNK) + sizeof(FramChunk::FRAM_CHUNK)
                + offsetof(FramCostChunk::FRAM_COST_CHUNK, SynchChannel);

        inline void extractDisplacements();
        inline void readFromFile();
        inline std::ifstream& selectFileAndOffset(int n, int& localFinalFrame, int& localInitialFrame,
                decltype(train.end())& fit, decltype(train.end())& next_fit, bool& veryFirst);
        inline void readTimestamp(std::ifstream& in, decltype(train.end())& fit, double*& local_traces);

    protected:
        double* traces;
        ProgressFunction progressFunction;
        std::string progressMessage;
        void* handle;

        virtual void clearState();

    public:
        explicit TraceReader(StreamFileTrain& train);
        TraceReader(const TraceReader& other) = delete;
        TraceReader(TraceReader&& other) noexcept;
        virtual ~TraceReader();

        TraceReader& operator=(const TraceReader& other) = delete;
        TraceReader& operator=(TraceReader&& other) noexcept;

        /**
         *
         * @return distance from the frame header to the arrival time value. Please, note that the size
         * of the arrival time is 64 bits.
         */
        static size_t getArrivalTimeDisplacement() { return ARRIVAL_TIME_DISPLACEMENT; }

        /**
         *
         * @return distance from the frame header to the value of synchronization channel 0. Please, note that
         * the size of the synchronization channel value is 32 bits, unsigned
         */
        static size_t getSynchronizationChannelDisplacement() { return SYNCHRONIZATION_CHANNEL_DISPLACEMENT; }

        /**
         *
         * @return the frame header size. Please, note that the size of a single pixel is 16 bits, unsigned
         */
        [[nodiscard]] size_t getFrameBodyDisplacement() const { return train.getFrameHeaderSize(); }

        /**
         *
         * @return Total number of traces already read
         */
        [[nodiscard]] int getChannelNumber() const { return pixelList.size(); }

        /**
         *
         * @return true if all traces have read by application of read() method
         */
        [[nodiscard]] bool hasRead() const { return _hasRead; }

        /**
         * Returns information about a certain channel. Please, be care, because the read() method
         * will change the channel order
         *
         * @param index The channel index
         * @return reference to the channel information
         */
        [[nodiscard]] const PixelListItem& getPixelItem(int index) const;

        /**
         * Returns channels for all indices
         *
         * @return a C style matrix where the column corresponds to the channel and the row corresponds to the
         * timestamp
         */
        [[nodiscard]] const double* getTraces() const;

        /**
         *
         * @return index of the initial frame
         */
        [[nodiscard]] int getInitialFrame() const { return initialFrame; }

        /**
         *
         * @return the final frame
         */
        [[nodiscard]] int getFinalFrame() const { return finalFrame; }

        /**
         *
         * @return total number of frames
         */
        [[nodiscard]] int getFrameNumber() const { return finalFrame - initialFrame + 1; }

        /**
         * Returns a certain value from the trace
         *
         * @param n timestamp number
         * @param idx trace index. Use getPixelItem to receive information about the index
         * @return the timestamp value
         */
        [[nodiscard]] double getValue(int n, int idx) const;

        [[nodiscard]] int getMapSizeX() const { return train.getXSize(); }
        [[nodiscard]] int getMapSizeY() const { return train.getYSize(); }
        [[nodiscard]] int getSynchChannelNumber() const { return train.getSynchronizationChannelNumber(); }

        /**
         * Sets the progress function
         *
         * @param f function handle. The function a handle pointer to will be called each time where 100 frames
         * will be processed
         * @param progress_message to be substituted to the function f
         * @param dlg a pointer that will be substituted to the last parameter of the function f
         */
        void setProgressFunction(ProgressFunction f, const std::string& progress_message, void* dlg){
            progressFunction = f;
            progressMessage = progress_message;
            handle = dlg;
        }

        /**
         * Prints all pixels one by one
         */
        void printAllPixels();

        /**
         * Adds a pixel to the pixel list
         *
         * @param pixel
         */
        void addPixel(const PixelListItem& pixel) { pixelList.push_back(pixel); }

        /**
         * Clears all pixels
         */
        void clearPixels() { pixelList.clear(); }

        /**
         *
         * @return the string "TRACE READER" if nothing else is defined in the derived class
         */
        [[nodiscard]] virtual const char* getReaderName() const  { return "TRACE READER"; }

        /**
         * Sets the frame range based on synchronization. Note, that the synchronization shall be completed
         *
         * @param sync the Synchronization object
         */
        void setFrameRange(Synchronization& sync);

        /**
         * Reads the traces
         */
        virtual void read();

        friend std::ostream& operator<<(std::ostream& out, const TraceReader& reader);

        class TraceReaderException: public iman_exception{
        public:
            TraceReaderException(const std::string& message): iman_exception(message) {};
        };

        class PixelItemIndexException: public TraceReaderException {
        public:
            PixelItemIndexException(): TraceReaderException("Channel index is out of bounds") {};
        };

        class TracesNotReadException: public TraceReaderException {
        public:
            TracesNotReadException(): TraceReaderException("Traces have not read") {};
        };

        class TimestampException: public TraceReaderException {
        public:
            TimestampException(): TraceReaderException("No timestamp with a given index") {};
        };

        class TraceNameException: public TraceReaderException{
        public:
            TraceNameException(): TraceReaderException("The tuple doesn't refer to a valid trace") {};
        };

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H
