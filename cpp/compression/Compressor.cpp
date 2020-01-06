//
// Created by serik1987 on 06.01.2020.
//

#include "Compressor.h"

namespace GLOBAL_NAMESPACE{

    std::string Compressor::getOutputFile(const std::string &input_file) {
        if (input_file.length() >= 16){
            return input_file;
        } else {
            return input_file + "z";
        }
    }
}