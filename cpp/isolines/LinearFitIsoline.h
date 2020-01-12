//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFITISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFITISOLINE_H

#include "Isoline.h"

namespace GLOBAL_NAMESPACE {

    class LinearFitIsoline: public Isoline {
    public:
        LinearFitIsoline(StreamFileTrain& train, Synchronization& sync): Isoline(train, sync) {};
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFITISOLINE_H
