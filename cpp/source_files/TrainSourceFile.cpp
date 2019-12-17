//
// Created by serik1987 on 17.12.2019.
//

#include "TrainSourceFile.h"
#include "SoftChunk.h"

namespace iman{

    void TrainSourceFile::loadFileInfo() {
        SourceFile::loadFileInfo();
        if (softChunk->getPreviousFilename() != "" && loadFileInfoMode == NotInHeadFail){
            throw not_train_head(this);
        }
        if (loadFileInfoMode == NotInHeadTraverse){
            while (softChunk->getPreviousFilename() != ""){
                setName(softChunk->getPreviousFilename());
                delete softChunk;
                softChunk = nullptr;
                close();
                open();
                SourceFile::loadFileInfo();
            }
        }
    }
}