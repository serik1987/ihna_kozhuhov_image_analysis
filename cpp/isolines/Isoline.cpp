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

    std::ostream &operator<<(std::ostream &out, const Isoline &isoline) {
        out << "===== ISOLINE =====\n";
        out << "Isoline name: " << isoline.getName() << "\n";
        out << "Analysis range: " << isoline.getAnalysisInitialCycle() << " - " << isoline.getAnalysisFinalCycle()
            << " cycles (" << isoline.getAnalysisInitialFrame() << "-" << isoline.getAnalysisFinalFrame() <<
            " frames)\n";
        out << "Isoline plotting range: "
            << isoline.getIsolineInitialCycle()
            << " - "
            << isoline.getIsolineFinalCycle()
            << " cycles ("
            << isoline.getIsolineInitialFrame()
            << "-"
            << isoline.getIsolineFinalFrame()
            << " frames)\n";
        isoline.printSpecial(out);

        return out;
    }

    void Isoline::clearState() {
        analysisInitialCycle = -1;
        analysisFinalCycle = -1;
        analysisInitialFrame = -1;
        analysisFinalFrame = -1;
        isolineInitialCycle = -1;
        isolineFinalCycle = -1;
        isolineInitialFrame = -1;
        isolineFinalFrame = -1;
    }

    void Isoline::synchronizeIsolines() {
        sync().synchronize();
        isolineInitialCycle = sync().getInitialCycle();
        isolineFinalCycle = sync().getFinalCycle();
        isolineInitialFrame = sync().getInitialFrame();
        isolineFinalFrame = sync().getFinalFrame();
    }

    void Isoline::synchronizeSignal() {
        sync().synchronize();
        analysisInitialCycle = sync().getInitialCycle();
        analysisInitialFrame = sync().getInitialFrame();
        analysisFinalCycle = sync().getFinalCycle();
        analysisFinalFrame = sync().getFinalFrame();
    }
}