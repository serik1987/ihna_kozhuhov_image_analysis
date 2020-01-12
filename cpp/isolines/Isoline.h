//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H

#include <functional>
#include "../synchronization/Synchronization.h"

namespace GLOBAL_NAMESPACE {

    /**
     * This is the base class for all objects that provide the isoline remove. This class is abstract, you can't
     * use it in anyway. However, you may use some of its derived classes each of which corresponds to a certain
     * algorithm to the isoline remove
     *
     * Use TraceReaderAndCleaner to apply this object for isoline remove from traces
     */
    class Isoline {
    private:
        Synchronization* psync;
        StreamFileTrain* ptrain;

    protected:
        int offset;

        Synchronization& sync() { return *psync; }
        StreamFileTrain& train() { return *ptrain; }

        ProgressFunction progressFunction;

    public:
        Isoline(StreamFileTrain& train, Synchronization& sync);
        Isoline(const Isoline& other);
        virtual ~Isoline() = default;

        Isoline& operator=(const Isoline& other) noexcept;

        int initial_frame() { return psync->getInitialFrame(); }
        int final_frame() { return psync->getFinalFrame(); }

        [[nodiscard]] virtual const char* getName() const noexcept = 0;
    };

}


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
