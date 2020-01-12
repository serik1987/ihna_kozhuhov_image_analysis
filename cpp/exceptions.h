//
// Created by serik1987 on 15.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_CORE_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_CORE_H

#include <string>
#include <exception>
#include "../compile_options.h"
#include "../init.h"

namespace GLOBAL_NAMESPACE {

    /**
     * When the class supports a progress bar, this runs so called "progress function" after
     * finishing processing of each 100 frames. The following input arguments will be passed:
     *  completed - total frames that have already been processed
     *  total - total number of frames.
     *  stage - pointer to the C-style string that contains a short description of a certain stage.
     *
     * if the function returns bool, the frame processing is completed. However, if the function returns false,
     * the processing will be aborted and the initial stage before the processing will be restored.
     */
    typedef bool (*ProgressFunction)(int completed, int total, const char* stage);

    class iman_exception : public std::exception {
    private:
        std::string message;

    public:
        explicit iman_exception(const std::string &msg) : message(msg) {};

        [[nodiscard]] const char *what() const noexcept override {
            return message.c_str();
        }
    };

    class io_exception : public iman_exception {
    private:
        std::string filename;

    public:
        io_exception(const std::string &message, const std::string &filename) :
                iman_exception(message + ": " + filename), filename(filename) {};

        [[nodiscard]] const std::string& getFilename() const { return filename; }
    };

}

#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_CORE_H
