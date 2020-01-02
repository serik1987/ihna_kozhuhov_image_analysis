//
// Created by serik1987 on 18.12.2019.
//

#include "FileTrain.h"
#include "SoftChunk.h"
#include "IsoiChunk.h"
#include "CostChunk.h"
#include "EpstChunk.h"
#include "Frame.h"


namespace GLOBAL_NAMESPACE{

    FileTrain::~FileTrain() {
        using namespace std;
#ifdef DEBUG_DELETE_CHECK
        std::cout << "TRAIN DELETE\n";
#endif
        for (auto pfile: *this){
            delete pfile;
        }
    }

    std::ostream &operator<<(std::ostream &out, const FileTrain &train) {
        using std::endl;

        out << "===== " << train.getFilename() << " =====\n";
        out << "Full path to the file train: " << train.getFilePath() << endl;
        out << "Number of files in the train: " << train.getFileNumber() << "\n";
        out << "Total number of frames: " << train.getTotalFrames() << "\n";
        out << "Data type: " << train.getDataType() << endl;
        out << "Frame size on X, pixels: " << train.getXSize() << endl;
        out << "Frame size on Y, pixels: " << train.getYSize() << endl;
        out << "Total number of pixels on the frame: " << train.getXYSize() << endl;
        out << "Frame header size, bytes: " << train.getFrameHeaderSize() << endl;
        out << "Frame body size, bytes: " << train.getFrameImageSize() << endl;
        out << "Frame size, bytes: " << train.getFrameSize() << endl;
        out << "File header size: " << train.getFileHeaderSize() << endl;
        if (train.getExperimentalMode() == FileTrain::Continuous){
            out << "Experimental mode: continuous\n";
            out << "Total number of synchronization channels: " << train.getSynchronizationChannelNumber() << endl;
            for (int chan = 0; chan < train.getSynchronizationChannelNumber(); ++chan){
                out << "Maximum signal value for synchronization channel " << chan << ": " <<
                    train.getSynchronizationChannelMax(chan) << " \n";
            }
        } else if (train.getExperimentalMode() == FileTrain::Episodic){
            out << "Experimental mode: episodic\n";
        } else {
            out << "Experimental mode: no idea\n";
        }
        out << "Train status: ";
        if (train.isOpened()){
            out << "opened";
        } else {
            out << "closed";
        }

        return out;
    }

    void FileTrain::open() {
        TrainSourceFile* file;

        if (isTraverse()){
            file = createFile(getFilePath(), getFilename(), TrainSourceFile::NotInHeadTraverse, getFilename());
        } else {
            file = createFile(getFilePath(), getFilename(), TrainSourceFile::NotInHeadFail, getFilename());
        }
        if (file == nullptr){
            throw std::runtime_error("FileTrain::createFile shall return non-nullptr value");
        }
        push_back(file);
        file->open();
        file->loadFileInfo();
        loadTrainProperties(*file);
        file->offsetFrame = 0;
        totalFrames = file->getSoftChunk().getFramesThisFile();

        while (file->getSoftChunk().getNextFilename() != ""){
            file = createFile(getFilePath(), file->getSoftChunk().getNextFilename(), TrainSourceFile::NotInHeadIgnore,
                    getFilename());
            if (file == nullptr){
                throw std::runtime_error("FileTrain::createFile shall return non-nullptr value");
            }
            push_back(file);
            file->open();
            file->loadFileInfo();
            file->offsetFrame = totalFrames;
            totalFrames += file->getSoftChunk().getFramesThisFile();
        }

        sanityCheck();

        opened = true;
    }

    void FileTrain::close() {
        for (auto pfile: *this){
            pfile->close();
        }
        opened = false;
    }

    void FileTrain::loadTrainProperties(TrainSourceFile &file) {
        filename = file.getFileName();
        frameHeaderSize = file.getFrameHeaderSize();
        fileHeaderSize = file.getFileHeaderSize();
        dataType = file.getSoftChunk().getDataType();
        xSize = file.getSoftChunk().getXSize();
        ySize = file.getSoftChunk().getYSize();
        xySize = xSize * ySize;
        frameImageSize = xySize * file.getSoftChunk().getDataTypeSize();
        frameSize = frameImageSize + frameHeaderSize;
        auto* cost = (CostChunk*)file.getIsoiChunk().getChunkById(ChunkHeader::COST_CHUNK_CODE);
        auto* epst = (EpstChunk*)file.getIsoiChunk().getChunkById(ChunkHeader::EPST_CHUNK_CODE);
        if (cost != nullptr && epst != nullptr){
            throw unsupported_experiment_mode_exception(this);
        }
        if (cost == nullptr && epst == nullptr){
            throw unsupported_experiment_mode_exception(this);
        }
        if (cost != nullptr){
            experimentalMode = Continuous;
            for (int chan = 0; chan < cost->getSynchronizationChannels(); ++chan){
                synchChannelMax.push_back(cost->getSynchronizationChannelsMax(chan));
            }
        }
        if (epst != nullptr){
            experimentalMode = Episodic;
        }
    }

    void FileTrain::sanityCheck() {
        using std::cout, std::endl;

        for (auto& file: *this){
            if (file->getSoftChunk().getTotalFrames() != getTotalFrames()){
                throw frame_number_mismatch(file);
            }
            if (file->getSoftChunk().getFramesThisFile() * getFrameSize() !=
                    file->getIsoiChunk().getChunkById(ChunkHeader::DATA_CHUNK_CODE)->getSize()){
                throw data_chunk_size_mismatch(file);
            }
            if (getFileSizeChecksum(*file) != 0){
                throw file_size_mismatch(file);
            }
            if (getDesiredIsoiChunkSize(*file) != file->getIsoiChunk().getSize()){
                throw isoi_chunk_size_mismatch(file);
            }

            if (getExperimentalMode() == Continuous){
                if (file->getIsoiChunk().getChunkById(ChunkHeader::COST_CHUNK_CODE) == nullptr){
                    throw experimental_chunk_not_found(file);
                }
            }
            if (getExperimentalMode() == Episodic){
                if (file->getIsoiChunk().getChunkById(ChunkHeader::EPST_CHUNK_CODE) == nullptr){
                    throw experimental_chunk_not_found(file);
                }
            }
            if (file->getFileHeaderSize() != getFileHeaderSize()){
                throw file_header_mismatch(file);
            }
            if (file->getFrameHeaderSize() != getFrameHeaderSize()){
                throw frame_header_mismatch(file);
            }
            if (file->getSoftChunk().getXSize() != getXSize() || file->getSoftChunk().getYSize() != getYSize()){
                throw map_dimensions_mismatch(file);
            }
            if (file->getSoftChunk().getDataType() != getDataType()){
                throw data_type_mismatch(file);
            }
        }
    }

    Frame *FileTrain::readFrame(int n) {
        auto* frame = new Frame(*this);
        try{
            auto& file = seek(n);
            frame->readFromFile(file, n);
        } catch (std::exception& e){
            delete frame;
            throw;
        }

        return frame;
    }
}
