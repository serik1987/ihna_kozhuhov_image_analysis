/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/tracereading/TraceReader.h"
#include "cpp/synchronization/NoSynchronization.h"
#include "cpp/synchronization/ExternalSynchronization.h"
#include "cpp/isolines/Isoline.h"
#include "cpp/isolines/NoIsoline.h"
#include "cpp/isolines/LinearFitIsoline.h"
#include "cpp/isolines/TimeAverageIsoline.h"

#define WORKING_DIR "/home/serik1987/vasomotor-oscillations/sample_data/c022z/"

using namespace ihna::kozhukhov::image_analysis;

int progress_function(int completed, int total){
    std::cout << (double)completed * 100 / total << "% completed\n";
}


int main() {
    using namespace std;

    {
        printf("Trace reading test...\n");
        const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};
        StreamFileTrain train(WORKING_DIR, "T_1BF.0200", file_size,
                              true);
        train.open();

        ExternalSynchronization sync1(train);
        NoSynchronization sync2(train);

        TimeAverageIsoline isoline1(train, sync1);
        std::cout << "C++ Isoline 1: " << isoline1.initial_frame() << " " << isoline1.final_frame() << std::endl;

        TimeAverageIsoline isoline2(train, sync2);
        std::cout << "C++ Isoline 2: " << isoline2.initial_frame() << " " << isoline2.final_frame() << std::endl;

        TimeAverageIsoline isoline3(isoline2);

        std::cout << "C++ Copy constructor was applied\n";
        std::cout << "C++ Isoline 1: " << isoline1.initial_frame() << " " << isoline1.final_frame() << std::endl;
        std::cout << "C++ Isoline 2: " << isoline2.initial_frame() << " " << isoline2.final_frame() << std::endl;
        std::cout << "C++ Isoline 3: " << isoline3.initial_frame() << " " << isoline3.final_frame() << std::endl;
        std::cout << "C++ Sync 1: " << sync1.getInitialFrame() << " " << sync1.getFinalFrame() << std::endl;
        std::cout << "C++ Sync 2: " << sync2.getFinalFrame() << " " << sync2.getFinalFrame() << std::endl;

        std::cout << "C++ Copy operator was applied\n";
        isoline1 = isoline3;
        std::cout << "C++ Isoline 1: " << isoline1.initial_frame() << " " << isoline1.final_frame() << std::endl;
        std::cout << "C++ Isoline 2: " << isoline2.initial_frame() << " " << isoline2.final_frame() << std::endl;
        std::cout << "C++ Isoline 3: " << isoline3.initial_frame() << " " << isoline3.final_frame() << std::endl;
        std::cout << "C++ Sync 1: " << sync1.getInitialFrame() << " " << sync1.getFinalFrame() << std::endl;
        std::cout << "C++ Sync 2: " << sync2.getFinalFrame() << " " << sync2.getFinalFrame() << std::endl;

    }
    return 0;
}
