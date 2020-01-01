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

    /*
    const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};
    StreamFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0200", file_size, true);
    train.open();
    Frame frame(train);

    cout << "Example of stream file" << endl;
    cout << train << endl;

    for (int n=0; n < train.getTotalFrames(); n += 1000){
        cout << "Reading frame number " << n << endl;
        TrainSourceFile& file = train.seek(n);
        ChunkHeader header = file.readChunkHeader();
        cout << header.getChunkId() << endl;
    }
    cout << "Reading frame number 9599\n";
    TrainSourceFile& file = train.seek(9599);
    ChunkHeader header = file.readChunkHeader();
    cout << header.getChunkId() << endl;
    */

    CompressedFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0A01z", true);
    train.open();
    Frame frame(train);

    cout << "Example of compressed file train" << endl;
    cout << train << endl;

    auto& file = train.seek(0);
    std::cout << file << std::endl;
    ChunkHeader header = file.readChunkHeader();
    cout << header.getChunkId() << endl;

    return 0;
}
