//
// Created by serik1987 on 17.12.2019.
//

#include "GreenSourceFile.h"

namespace ihna::kozhukhov::image_analysis{

    void GreenSourceFile::loadFileInfo(){
        SourceFile::loadFileInfo();
        if (getFileType() != GreenFile){
            throw not_green_file_exception(this);
        }
    }

}