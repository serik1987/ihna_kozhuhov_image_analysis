//
// Created by serik1987 on 11.01.2020.
//

#include <algorithm>
#include "TraceReader.h"
#include "../source_files/SoftChunk.h"


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
        if (_hasRead && traces != nullptr){
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

    void TraceReader::clearState(){
        initialDisplacement = 0;
        delete [] dataDisplacements;
        dataDisplacements = nullptr;
        delete [] dataTypes;
        dataTypes = nullptr;
        delete [] traces;
        traces = nullptr;
    }

    void TraceReader::read() {

        clearState();
        std::sort(pixelList.begin(), pixelList.end());
        auto last_item = std::unique(pixelList.begin(), pixelList.end());
        if (last_item != pixelList.end()){
            pixelList.erase(last_item, pixelList.end());
        }
        extractDisplacements();

        readFromFile();

        _hasRead = true;

        printf("\n");
        printf("C++: reading traces\n");
    }

    void TraceReader::extractDisplacements() {
        auto it = pixelList.begin();
        auto first = it;
        initialDisplacement = it->getDisplacement();
        dataTypes = new PixelListItem::PointType[pixelList.size()];
        dataDisplacements = new size_t[pixelList.size()];
        ++it;
        int idx = 0;
        auto current_it = pixelList.begin();
        for (; it != pixelList.end(); ++current_it, ++it, ++idx){
            dataTypes[idx] = current_it->getPointType();
            dataDisplacements[idx] = it->getDisplacement() - current_it->getDisplacement() - current_it->getPointSize();
        }
        dataTypes[idx] = current_it->getPointType();
        dataDisplacements[idx] = train.getFrameSize() - current_it->getDisplacement() - current_it->getPointSize() +
                first->getDisplacement();
    }

    void TraceReader::readFromFile() {
        std::ofstream f;
        f.open("/home/serik1987/vasomotor-oscillations/displacements.log", std::ios_base::out);
        if (f.fail()){
            throw std::runtime_error("Failed to open the test file!");
        }
        double* local_traces = traces = new double[getChannelNumber() * getFrameNumber()];
        auto next_fit = train.begin(); ++next_fit;
        auto fit = train.begin();
        int localInitialFrame = 0;
        int localFinalFrame = (*next_fit)->offsetFrame - 1;
        bool veryFirst = true;
        for (int n = initialFrame; n <= finalFrame; ++n){
            f << "n = " << n << "\t";
            uint32_t offset;
            if (n > localFinalFrame) {
                ++fit, ++next_fit;
                localInitialFrame = (*fit)->offsetFrame;
                if (next_fit == train.end()) {
                    localFinalFrame = train.getTotalFrames() - 1;
                } else {
                    localFinalFrame = (*next_fit)->offsetFrame - 1;
                }
                veryFirst = true;
            }
            if (veryFirst){
                int m = n - localInitialFrame;
                offset = train.getFileHeaderSize() + m * train.getFrameSize() + initialDisplacement;
                (*fit)->getFileStream().seekg(offset, std::ios_base::beg);
                if ((*fit)->getFileStream().fail()){
                    throw SourceFile::file_read_exception(*fit);
                }
                veryFirst = false;
            } else {
                offset = dataDisplacements[getChannelNumber()-1];
                (*fit)->getFileStream().seekg(offset, std::ios_base::cur);
                if ((*fit)->getFileStream().fail()){
                    throw SourceFile::file_read_exception(*fit);
                }
            }

            std::ifstream& file = (*fit)->getFileStream();
            for (int chan = 0; chan < getChannelNumber(); ++chan){
                if (dataTypes[chan] == PixelListItem::ArrivalTime){
                    uint64_t pix;
                    file.read((char*)&pix, 8);
                    if (file.fail()){
                        throw SourceFile::file_read_exception(*fit);
                    }
                    *(local_traces++) = pix;
                    f << pix << "\t";
                } else if (dataTypes[chan] == PixelListItem::SynchronizationChannel){
                    uint32_t pix;
                    file.read((char*)&pix, 4);
                    if (file.fail()){
                        throw SourceFile::file_read_exception(*fit);
                    }
                    *(local_traces++) = pix;
                    f << pix << "\t";
                } else if (dataTypes[chan] == PixelListItem::PixelValue){
                    uint16_t pix;
                    file.read((char*)&pix, 2);
                    if (file.fail()){
                        throw SourceFile::file_read_exception(*fit);
                    }
                    *(local_traces++) = pix;
                    f << pix << "\t";
                }
                if (chan != getChannelNumber() - 1){
                    file.seekg(dataDisplacements[chan], std::ios_base::cur);
                    if (file.fail()){
                        throw SourceFile::file_read_exception(*fit);
                    }
                }
            }

            f << std::endl;
        }
        f.close();
    }


}