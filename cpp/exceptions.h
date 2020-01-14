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
     * finishing processing of each 100 frames. Its main role is to tell the user how long he shall wait, how
     * much frames have been processed and how much frames are going to process
     * The following input arguments will be passed by the frame processing object to the function:
     *  completed - total frames that have already been processed
     *  total - total number of frames.
     *  stage - pointer to the C-style string that contains a short description of a certain stage.
     *  handle - the pointer that was passed during the call of setProgressFunction method. An exact value of this
     *  pointer will be passed to this function
     *
     * The function shall return to the processing object one of the following values:
     * true - the object has been processed. This is OK
     * false - the function has not been processed. The process shall be interrupted
     */
    typedef bool (*ProgressFunction)(int completed, int total, const char* stage, void* handle);

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
