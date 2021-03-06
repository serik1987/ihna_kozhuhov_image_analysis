//
// Created by serik1987 on 17.12.2019.
//

#ifndef IHNA_KOZKUKHOV_IMAGE_ANALYSIS_GREENSOURCEFILE_H
#define IHNA_KOZKUKHOV_IMAGE_ANALYSIS_GREENSOURCEFILE_H

#include "SourceFile.h"

namespace GLOBAL_NAMESPACE {

    class GreenSourceFile: public SourceFile {
    public:
        /**
         * Creates the source file that loads/stores green images
         *
         * @param path full path to the green image file
         * @param filename filename of the green image file
         */
        GreenSourceFile(const std::string& path, const std::string& filename):
            SourceFile(path, filename) {};

        class not_green_file_exception: public source_file_exception{
        public:
            explicit not_green_file_exception(GreenSourceFile* parent):
                source_file_exception(MSG_GREEN_FILE_EXCEPTION, parent) {};
            explicit not_green_file_exception(const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_GREEN_FILE_EXCEPTION, filename, trainname) {};
        };

        void loadFileInfo() override;

        /**
         *
         * @return short description of the desired file type
         */
        [[nodiscard]] std::string getFileTypeDescription() const override { return "Green file"; }
    };

}


#endif //IHNA_KOZKUKHOV_IMAGE_ANALYSIS_GREENSOURCEFILE_H
