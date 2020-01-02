/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/source_files/CompressedFileTrain.h"
#include "cpp/source_files/Frame.h"
#include "cpp/source_files/SoftChunk.h"

int main() {
    using namespace std;
    using namespace ihna::kozhukhov::image_analysis;

    const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};
    StreamFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0200", file_size, true);
    train.open();

    cout << "Example of stream file" << endl;
    cout << train << endl;

    for (int n = 0; n < 11; ++n){
        auto* frame = train.readFrame(n);
        cout << *frame << endl;
        delete frame;
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
