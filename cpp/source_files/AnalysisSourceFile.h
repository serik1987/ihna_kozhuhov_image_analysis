//
// Created by serik1987 on 17.12.2019.
//

#ifndef IHNA_KOZKUKHOV_IMAGE_ANALYSIS_ANALYSISSOURCEFILE_H
#define IHNA_KOZKUKHOV_IMAGE_ANALYSIS_ANALYSISSOURCEFILE_H

#include "SourceFile.h"

namespace ihna::kozhukhov::image_analysis {

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
                source_file_exception("The file is not an analysis source file", parent) {};
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
