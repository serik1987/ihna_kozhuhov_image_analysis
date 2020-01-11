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


int main() {
    using namespace std;

    {
        printf("Trace reading test...\n");
        const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};
        StreamFileTrain train(WORKING_DIR, "T_1BF.0200", file_size,
                              true);
        train.open();

        TraceReader reader(train);
        PixelListItem pixel1(reader, PixelListItem::ARRIVAL_TIME, 0);
        PixelListItem pixel2(reader, PixelListItem::SYNCH, 0);
        PixelListItem pixel3(reader, PixelListItem::SYNCH, 1);
        PixelListItem pixel4(reader, 0, 0);
        PixelListItem pixel5(reader, 0, 1);
        PixelListItem pixel6(reader, 1, 0);

        cout << pixel1 << endl;
        cout << pixel2 << endl;
        cout << pixel3 << endl;
        cout << pixel4 << endl;
        cout << pixel5 << endl;
        cout << pixel6 << endl;
    }
    return 0;
}
