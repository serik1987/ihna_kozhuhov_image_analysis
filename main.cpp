/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/tracereading/TraceReader.h"

#define WORKING_DIR "/home/serik1987/vasomotor-oscillations/sample_data/c022z/"

using namespace ihna::kozhukhov::image_analysis;

int progress_function(int completed, int total){
    std::cout << (double)completed * 100 / total << "% completed\n";
}

TraceReader get_trace_reader(StreamFileTrain& train){
    TraceReader reader(train);

    return reader;
}


int main() {
    using namespace std;

    {
        printf("Trace reading test...\n");
        const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};
        StreamFileTrain train(WORKING_DIR, "T_1BF.0200", file_size,
                              true);
        train.open();

        TraceReader reader1(train);
        reader1 = get_trace_reader(train);
    }
    return 0;
}
