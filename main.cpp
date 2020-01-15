/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/synchronization/ExternalSynchronization.h"

#define WORKING_DIR "/home/serik1987/vasomotor-oscillations/sample_data/c022z/"

using namespace ihna::kozhukhov::image_analysis;

bool progress_function(int completed, int total, const char* msg, void* handle){
    std::cout << (double)completed * 100 / total << "% completed\n";
}


int main() {
    using namespace std;

    {
        printf("C++ Trace reading test...\n");
        const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};
        StreamFileTrain train(WORKING_DIR, "T_1BF.0200", file_size,
                              true);
        train.open();

        ExternalSynchronization sync(train);

        sync.setSynchronizationChannel(1);
        sync.setInitialCycle(2);
        sync.setFinalCycle(10);
        sync.setDoPrecise(true);
        sync.setHarmonic(2.0);

        cout << "Initial frame: " << sync.getInitialFrame() << endl;
        cout << "Final frame: " << sync.getFinalFrame() << endl;
        cout << "Frame number: " << sync.getFrameNumber() << endl;
        cout << "Precise: " << sync.isDoPrecise() << endl;
        cout << "Phase increment: " << sync.getPhaseIncrement() << endl;
        cout << "Initial phase: " << sync.getInitialPhase() << endl;
        cout << "Harmonic: " << sync.getHarmonic() << endl;
        cout << "Synchronization channel: " << sync.getSynchronizationChannel() << endl;
        cout << "Initial cycle: " << sync.getInitialCycle() << endl;
        cout << "Final cycle: " << sync.getFinalCycle() << endl;

    }
    return 0;
}
