/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/source_files/CompressedFileTrain.h"

#include "cpp/compression/Compressor.h"
#include "cpp/compression/Decompressor.h"

#define WORKING_DIR "/home/serik1987/vasomotor-oscillations/sample_data/c022z/"

void progress_function(float perc){
    std::cout << perc << "% completed\n";
}


int main() {
    using namespace std;
    using namespace ihna::kozhukhov::image_analysis;

    {
        printf("Compression test...\n");
        const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 380726636};
        StreamFileTrain train(WORKING_DIR, "T_1BF.0A00", file_size,
                              true);
        train.open();

        Compressor compressor(train, WORKING_DIR);
        compressor.setProgressFunction(progress_function);
        compressor.run();
    }
    /*
    {
        printf("Decompression test...\n");
        CompressedFileTrain train(WORKING_DIR, "T_1BF.0A00z", true);
        train.open();

        Decompressor decompressor(train, WORKING_DIR);
        decompressor.setProgressFunction(progress_function);
        decompressor.run();
    }
    */
    return 0;
}
