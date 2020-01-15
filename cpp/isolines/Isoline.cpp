//
// Created by serik1987 on 12.01.2020.
//

#include "Isoline.h"


namespace GLOBAL_NAMESPACE {

    Isoline::Isoline(StreamFileTrain &train, Synchronization &sync): ptrain(&train), psync(&sync) {
        offset = 0;
        analysisInitialCycle = -1;
        analysisFinalCycle = -1;
        isolineInitialCycle = -1;
        isolineFinalCycle = -1;
        analysisInitialFrame = -1;
        analysisFinalFrame = -1;
        isolineInitialFrame = -1;
        isolineFinalFrame = -1;
    }

    Isoline::Isoline(const Isoline &other): ptrain(other.ptrain), psync(other.psync) {
        offset = other.offset;
    }

    Isoline &Isoline::operator=(const Isoline &other) noexcept {
        ptrain = other.ptrain;
        psync = other.psync;
        offset = other.offset;

        return *this;
    }
}