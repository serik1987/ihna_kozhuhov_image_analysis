//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H

#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This synchronization is based on assumption that the stimulus is continuous and its period equals to
     * integer number of frames and expresses in frame number
     * 1) The synchronization signal is the timestamp number multiplied by 2PI and divided by the stimulus period
     * 2) initial and final frame is defined in such a way as to contain integer number of cycles. The first cycle
     * is assumed to start at the beginning of the record
     */
    class QuasiStimulusSynchronization: public Synchronization {
    private:
        int stimulusPeriod;
        int initialCycle;
        int finalCycle;

    public:
        explicit QuasiStimulusSynchronization(StreamFileTrain& other);
        QuasiStimulusSynchronization(const QuasiStimulusSynchronization& other) = delete;
        QuasiStimulusSynchronization(QuasiStimulusSynchronization&& other) noexcept;

        QuasiStimulusSynchronization& operator=(const QuasiStimulusSynchronization& other) = delete;
        QuasiStimulusSynchronization& operator=(QuasiStimulusSynchronization&& other) noexcept;

        [[nodiscard]] const char* getName() const noexcept override {return "quasi-stimulus synchronization"; }
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H
