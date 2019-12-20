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
