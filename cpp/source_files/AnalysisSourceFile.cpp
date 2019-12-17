//
// Created by serik1987 on 17.12.2019.
//

#include "AnalysisSourceFile.h"

namespace iman{

    void AnalysisSourceFile::loadFileInfo(){
        SourceFile::loadFileInfo();
        if (getFileType() != AnalysisFile){
            throw not_analysis_file_exception(this);
        }
    }

}