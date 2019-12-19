/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/source_files/IsoiChunk.h"
#include "cpp/source_files/SoftChunk.h"

int main() {
    using namespace std;
    using namespace ihna::kozhukhov::image_analysis;

    const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};

    StreamFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0200", file_size, true);
    train.open();

    return 0;
}
