//
// Created by serik1987 on 18.12.2019.
//

#include "FileTrain.h"
#include "SoftChunk.h"


namespace iman{

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
            out << "Total number of synchronization channels: " << train.getSynchnorizationChannelNumber() << endl;
            for (int chan = 0; chan < train.getSynchnorizationChannelNumber(); ++chan){
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
            out << "opened\n";
        } else {
            out << "closed\n";
        }
        out << "Number of files in the train: " << train.getFileNumber();

        return out;
    }

    void FileTrain::open() {
        TrainSourceFile* file;

        if (isTraverse()){
            file = createFile(getFilePath(), getFilename(), TrainSourceFile::NotInHeadTraverse);
        } else {
            file = createFile(getFilePath(), getFilename(), TrainSourceFile::NotInHeadFail);
        }
        if (file == nullptr){
            throw std::runtime_error("FileTrain::createFile shall return non-nullptr value");
        }
        push_back(file);
        file->open();
        file->loadFileInfo();
        loadTrainProperties(*file);

        while (file->getSoftChunk().getNextFilename() != ""){
            file = createFile(getFilePath(), file->getSoftChunk().getNextFilename(), TrainSourceFile::NotInHeadIgnore);
            push_back(file);
            file->open();
            file->loadFileInfo();

        }

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
    }
}
