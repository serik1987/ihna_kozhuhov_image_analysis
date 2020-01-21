//
// Created by serik1987 on 12.01.2020.
//

#include "LinearFitIsoline.h"

namespace GLOBAL_NAMESPACE {

    void LinearFitIsoline::traceCleaning(TraceReaderAndCleaner &reader, const double *src, const double *srcLeft,
                                         const double *srcRight, double *isolines) {
        std::cout << "Linear fit isolines: trace cleaning\n";
        std::cout << "src = " << src << std::endl;
        std::cout << "srcLeft = " << srcLeft << std::endl;
        std::cout << "srcRight = " << srcRight << std::endl;
        std::cout << "isolines = " << isolines << std::endl;
    }
}