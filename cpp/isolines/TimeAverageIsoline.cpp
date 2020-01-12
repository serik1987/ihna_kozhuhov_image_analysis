//
// Created by serik1987 on 12.01.2020.
//

#include "TimeAverageIsoline.h"

namespace GLOBAL_NAMESPACE {

    TimeAverageIsoline::TimeAverageIsoline(StreamFileTrain &train, Synchronization &sync) : Isoline(train, sync) {
        averageCycles = 1;
    }

    TimeAverageIsoline::TimeAverageIsoline(const TimeAverageIsoline &other): Isoline(other) {
        averageCycles = other.averageCycles;
    }

    TimeAverageIsoline &TimeAverageIsoline::operator=(const TimeAverageIsoline &other) {
        Isoline::operator=(other);
        averageCycles = other.averageCycles;

        return *this;
    }
}