//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H

#include "Synchronization.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This synchronization is based on so called "synchronization channel" that contains signal received from the
     * stimulus synchronizer.
     */
    class ExternalSynchronization: public Synchronization {
    private:
        int synchronizationChannel;
        double* synchronizationSignal;

        int initialCycle;
        int finalCycle;

    public:
        explicit ExternalSynchronization(StreamFileTrain& train);
        ExternalSynchronization(const ExternalSynchronization& other) = delete;
        ExternalSynchronization(ExternalSynchronization&& other) noexcept;
        ~ExternalSynchronization() override;

        ExternalSynchronization& operator=(const ExternalSynchronization& other) = delete;
        ExternalSynchronization& operator=(ExternalSynchronization&& other) noexcept;

        [[nodiscard]] const char* getName() const noexcept override { return "external synchronization"; }

    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H
