//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H

#include <functional>
#include "../synchronization/Synchronization.h"

namespace GLOBAL_NAMESPACE {

    class Isoline {
    private:
        Synchronization* psync;
        StreamFileTrain* ptrain;

    protected:
        int offset;

        Synchronization& sync() { return *psync; }
        StreamFileTrain& train() { return *ptrain; }

    public:
        Isoline(StreamFileTrain& train, Synchronization& sync);
        Isoline(const Isoline& other);
        virtual ~Isoline() = default;

        Isoline& operator=(const Isoline& other) noexcept;

        int initial_frame() { return psync->getInitialFrame(); }
        int final_frame() { return psync->getFinalFrame(); }
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
