/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#define PRINT_CACHE_ROUTINES

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/source_files/CompressedFileTrain.h"
#include "cpp/source_files/Frame.h"
#include "cpp/source_files/SoftChunk.h"

#ifdef PRINT_CACHE_ROUTINES

namespace GLOBAL_NAMESPACE {

    void print_cache(GLOBAL_NAMESPACE::FileTrain *train) {
        using namespace std;
        cout << "Frame cache: ";
        for (auto* frame: train->frameCache){
            cout << frame->getFrameNumber() << " ";
        }
        cout << "\n";

        if (train->frameCacheStatus != nullptr){
            cout << "Frame cache status: ";
            for (int i=0; i < train->getTotalFrames(); ++i){
                if (train->frameCacheStatus[i] != nullptr){
                    cout << i << " ";
                }
            }
            cout << endl;
        } else {
            cout << "Frame cache status is not defined\n";
        }
    }

}

#endif

int main() {
    using namespace std;
    using namespace ihna::kozhukhov::image_analysis;

    const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};
    StreamFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0200", file_size, true);
    train.open();

    cout << "Example of stream file" << endl;
    cout << train << endl;

    print_cache(&train);
    train.capacity = 10;

    for (int n = 0; n < 30; ++n){
        Frame& frame = train[n];
        print_cache(&train);
        cout << "t = " << frame.getFramChunk().getTimeArrival() << endl;
    }

    /*
    CompressedFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0A01z", true);
    train.open();
    Frame frame(train);

    cout << "Example of compressed file train" << endl;
    cout << train << endl;

    auto& file = train.seek(0);
    frame.readFromFile(file, 0);
    std::cout << file << std::endl;
    std::cout << frame << std::endl;
    */

    return 0;
}
