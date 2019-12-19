//
// Created by serik1987 on 17.12.2019.
//

#include "TrainSourceFile.h"
#include "SoftChunk.h"
#include "IsoiChunk.h"

namespace ihna::kozhukhov::image_analysis{

    void TrainSourceFile::loadFileInfo() {
        SourceFile::loadFileInfo();
        if (softChunk->getPreviousFilename() != "" && loadFileInfoMode == NotInHeadFail){
            throw not_train_head(this);
        }
        if (loadFileInfoMode == NotInHeadTraverse){
            while (softChunk->getPreviousFilename() != ""){
                setName(softChunk->getPreviousFilename());
                delete softChunk;
                delete isoiChunk;
                softChunk = nullptr;
                isoiChunk = nullptr;
                close();
                open();
                SourceFile::loadFileInfo();
            }
        }
        isoiChunk->readFromFile(*this);
    }
}