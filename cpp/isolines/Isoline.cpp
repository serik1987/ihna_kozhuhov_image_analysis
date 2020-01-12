//
// Created by serik1987 on 12.01.2020.
//

#include "Isoline.h"


namespace GLOBAL_NAMESPACE {

    Isoline::Isoline(StreamFileTrain &train, Synchronization &sync): ptrain(&train), psync(&sync) {
        offset = 0;
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