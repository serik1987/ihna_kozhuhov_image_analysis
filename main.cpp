/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/source_files/Frame.h"

int main() {
    using namespace std;
    using namespace ihna::kozhukhov::image_analysis;

    const uint32_t file_size[] = {1299503468, 1299503468, 1299503468, 1135885676};

    StreamFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0201", file_size, true);
    train.open();
    Frame frame(train);

    std::cout << frame << std::endl;

    return 0;
}
