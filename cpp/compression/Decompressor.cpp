//
// Created by serik1987 on 06.01.2020.
//

#include "Decompressor.h"

namespace GLOBAL_NAMESPACE{

    std::string Decompressor::getOutputFile(const std::string &input_file) {
        auto S = input_file.length();
        if (input_file[S-1] == 'z'){
            return input_file.substr(0, S-1);
        } else {
            return input_file;
        }
    }
}