//
// Created by serik1987 on 11.01.2020.
//

#include "TraceReader.h"


namespace GLOBAL_NAMESPACE {

    TraceReader::TraceReader(StreamFileTrain &train): train(train), pixelList() {
        initialFrame = 0;
        finalFrame = train.getTotalFrames()-1;

        initialDisplacement = 0;
        dataDisplacements = nullptr;
        dataTypes = nullptr;
        _hasRead = false;

        traces = nullptr;
    }

    TraceReader::TraceReader(TraceReader &&other) noexcept: train(other.train), pixelList(other.pixelList) {
        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;

        initialDisplacement = other.initialDisplacement;
        dataDisplacements = other.dataDisplacements;
        other.dataDisplacements = nullptr;
        dataTypes = other.dataTypes;
        other.dataTypes = nullptr;
        _hasRead = other._hasRead;

        traces = other.traces;
        other.traces = nullptr;
    }

    TraceReader::~TraceReader() {
#ifdef DEBUG_DELETE_CHECK
        std::cout << "DELETE TRACE READER\n";
#endif
        delete [] dataDisplacements;
        delete [] dataTypes;
        delete [] traces;
    }

    TraceReader &TraceReader::operator=(TraceReader &&other) noexcept {
        train = other.train;

        initialFrame = other.initialFrame;
        finalFrame = other.finalFrame;

        initialDisplacement = other.initialDisplacement;
        dataDisplacements = other.dataDisplacements;
        other.dataDisplacements = nullptr;
        dataTypes = other.dataTypes;
        other.dataTypes = nullptr;
        _hasRead = other._hasRead;

        traces = other.traces;
        other.traces = nullptr;

        return *this;
    }

    const PixelListItem& TraceReader::getPixelItem(int index) const {
        if (index < pixelList.size() && index >= 0){
            return pixelList[index];
        } else {
            throw PixelItemIndexException();
        }
    }

    const double *TraceReader::getTraces() const {
        if (_hasRead){
            return traces;
        } else {
            throw TracesNotReadException();
        }
    }

    double TraceReader::getValue(int n, int idx) const {
        if (n < 0 || n >= finalFrame - initialFrame){
            throw TimestampException();
        }
        if (idx < 0 || idx >= getChannelNumber()){
            throw PixelItemIndexException();
        }
        if (!_hasRead){
            throw TracesNotReadException();
        }
        return traces[n * getChannelNumber() + idx];
    }

    std::ostream &operator<<(std::ostream &out, const TraceReader &reader) {
        using namespace std;

        out << "===== " << reader.getReaderName() << " =====" << endl;
        out << "Frame arrival time: 0x" << hex << reader.getArrivalTimeDisplacement() << dec << endl;
        out << "Synchronization channel displacement: 0x" << hex << reader.getSynchronizationChannelDisplacement() <<
             dec << endl;
        out << "Frame body displacement: 0x" << hex << reader.getFrameBodyDisplacement() << dec << endl;
        if (reader.hasRead()) {
            out << "The channel has been read\n";
        } else {
            out << "The channel has not been read\n";
        }
        out << "Number of trace channels: " << reader.getChannelNumber() << endl;
        out << "Initial frame: " << reader.getInitialFrame() << endl;
        out << "Final frame: " << reader.getFinalFrame() << endl;
        out << "Total frame number: " << reader.getFrameNumber() << endl;

        return out;
    }

    void TraceReader::printAllPixels() {
        for (auto& item: pixelList){
            std::cout << item;
        }
    }


}