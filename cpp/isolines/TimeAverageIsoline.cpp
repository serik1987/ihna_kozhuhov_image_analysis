//
// Created by serik1987 on 12.01.2020.
//

#include "TimeAverageIsoline.h"
#include "../synchronization/ExternalSynchronization.h"

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

    void TimeAverageIsoline::setAverageCycles(int r) {
        if (r > 0){
            averageCycles = r;
        } else {
            throw AverageCyclesException();
        }
    }

    void TimeAverageIsoline::printSpecial(std::ostream &out) const {
        out << "Average radius, cycles: " << getAverageCycles() << "\n";
    }

    void TimeAverageIsoline::extendRange() {
        isolineInitialCycle = sync().getInitialCycle();
        if (isolineInitialCycle != -1){
            isolineInitialCycle -= averageCycles;
            sync().setInitialCycle(isolineInitialCycle);
        }
        isolineFinalCycle = sync().getFinalCycle();
        if (isolineFinalCycle != -1){
            isolineFinalCycle += averageCycles;
            sync().setFinalCycle(isolineFinalCycle);
        }
    }

    void TimeAverageIsoline::sacrifice() {
        if (sync().getCycleNumber() <= 2 * averageCycles){
            throw ExternalSynchronization::TooFewFramesException();
        }
        std::cout << isolineInitialCycle + averageCycles << std::endl;
        sync().setInitialCycle(isolineInitialCycle + averageCycles);
        sync().setFinalCycle(isolineFinalCycle - averageCycles);
    }

    void TimeAverageIsoline::traceCleaning(TraceReaderAndCleaner &reader, const double *src, const double *srcLeft,
                                           const double *srcRight, double *isolines) {
        std::cout << "Time average: trace cleaning\n";
        std::cout << "src = " << src << std::endl;
        std::cout << "srcLeft = " << srcLeft << std::endl;
        std::cout << "srcRight = " << srcRight << std::endl;
        std::cout << "isolines = " << isolines << std::endl;
    }
}