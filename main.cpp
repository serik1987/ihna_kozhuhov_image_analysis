/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/synchronization/ExternalSynchronization.h"
#include "cpp/isolines/TimeAverageIsoline.h"
#include "cpp/tracereading/TraceReaderAndCleaner.h"

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
        TimeAverageIsoline isoline(train, sync);
        TraceReaderAndCleaner reader(train);
        reader.setIsolineRemover(isoline);

        cout << "Frame offset: " << isoline.getFrameOffset() << endl;
        cout << "Analysis epoch (cycles): " << isoline.getAnalysisInitialCycle() << " " << isoline.getAnalysisFinalCycle() << "\n";
        cout << "Analysis epoch (frames): " << isoline.getAnalysisInitialFrame() << " " << isoline.getAnalysisFinalFrame() << "\n";
        cout << "Time average epoch (cycles): " << isoline.getIsolineInitialCycle() << " " << isoline.getIsolineFinalCycle() << "\n";
        cout << "Time average epoch (frames): " << isoline.getIsolineInitialFrame() << " " << isoline.getIsolineFinalFrame() << "\n";
        cout << "Time average radius (cycles): " << isoline.getAverageCycles() << "\n";

        cout << reader << endl;

        reader.addPixel(PixelListItem(reader, PixelListItem::ARRIVAL_TIME, 0));
        reader.addPixel(PixelListItem(reader, PixelListItem::SYNCH, 0));
        reader.addPixel(PixelListItem(reader, PixelListItem::SYNCH, 1));

        for (int i=100; i < 200; i++){
            for (int j=100; j < 200; j++){
                reader.addPixel(PixelListItem(reader, i, j));
            }
        }

        reader.clearPixels();

        reader.printAllPixels();

    }
    return 0;
}
