/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/synchronization/ExternalSynchronization.h"
#include "cpp/isolines/TimeAverageIsoline.h"
#include "cpp/accumulators/MapFilter.h"

#define WORKING_DIR "/home/serik1987/vasomotor-oscillations/c022/"

using namespace ihna::kozhukhov::image_analysis;

bool progress_function(int completed, int total, const char* msg, void* handle){
    std::cout << msg << std::endl;
    std::cout << handle << std::endl;
    std::cout << (double)completed * 100 / total << "% completed\n";
}


int main() {
    using namespace std;

    {
        printf("CPP Trace reading test...\n");
        const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 380726636};
        StreamFileTrain train(WORKING_DIR, "T_1BF.0100", file_size,
                              true);
        train.open();

        ExternalSynchronization sync(train);
        TimeAverageIsoline isoline(train, sync);
        MapFilter filter(isoline);
        vector<double> b = {1, 2, 3, 4};
        vector<double> a = {5, 6, 7, 8};
        filter.setB(b);
        filter.setA(a);

        cout << filter << endl;
        cout << "CPP Isoline" << endl;
        cout << filter.getIsoline() << endl;
        cout << "CPP Synchronization" << endl;
        cout << filter.getSynchronization() << endl;
        cout << "CPP Get train" << endl;
        cout << filter.getTrain() << endl;

        filter.accumulate();

    }
    return 0;
}
