//
// Created by serik1987 on 17.12.2019.
//

#ifndef IHNA_KOZKUKHOV_IMAGE_ANALYSIS_ANALYSISSOURCEFILE_H
#define IHNA_KOZKUKHOV_IMAGE_ANALYSIS_ANALYSISSOURCEFILE_H

#include "../../init.h"
#include "SourceFile.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This is the base class for IMAN source file containing analysis results
     */
    class AnalysisSourceFile: public SourceFile {
    public:
        /**
         * Creates new analysis file
         *
         * @param path the file path (including trailing '/' or '\' symbol)
         * @param filename the file name
         */
        AnalysisSourceFile(const std::string& path, const std::string& filename):
            SourceFile(path, filename) {};

        class not_analysis_file_exception: public source_file_exception{
        public:
            explicit not_analysis_file_exception(AnalysisSourceFile* parent):
                source_file_exception(MSG_NOT_ANALYSIS_FILE_EXCEPTION, parent) {};
            explicit not_analysis_file_exception(const std::string& filename, const std::string& trainname = ""):
                source_file_exception(MSG_NOT_ANALYSIS_FILE_EXCEPTION, filename, trainname) {};
        };

        /**
         * Loads the file information
         */
        void loadFileInfo() override;

        /**
         *
         * @return short description of the desired file type
         */
        [[nodiscard]] std::string getFileTypeDescription() const override {
            return "Analysis file";
        }
    };

}


#endif //IHNA_KOZKUKHOV_IMAGE_ANALYSIS_ANALYSISSOURCEFILE_H
