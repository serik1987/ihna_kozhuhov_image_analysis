//
// Created by serik1987 on 17.12.2019.
//

#include "CompressedSourceFile.h"

namespace ihna::kozhukhov::image_analysis{

    void CompressedSourceFile::loadFileInfo() {
        TrainSourceFile::loadFileInfo();
        if (getFileType() != CompressedFile){
            throw not_compressed_file_exception(this);
        }
    }
}