//
// Created by serik1987 on 17.12.2019.
//

#include "StreamSourceFile.h"

namespace GLOBAL_NAMESPACE{

    void StreamSourceFile::loadFileInfo() {
        TrainSourceFile::loadFileInfo();
        if (getFileType() != StreamFile){
            throw not_stream_file(this);
        }
    }
}