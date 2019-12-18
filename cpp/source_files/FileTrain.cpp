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
        out << "Frame header size: " << train.getFrameHeaderSize() << endl;
        out << "File header size: " << train.getFileHeaderSize() << endl;
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
        push_back(file);
        file->open();
        file->loadFileInfo();
        filename = file->getFileName();
        frameHeaderSize = file->getFrameHeaderSize();
        fileHeaderSize = file->getFileHeaderSize();

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
}
