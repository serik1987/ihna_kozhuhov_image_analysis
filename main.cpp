/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.image_analysis
 */

#define DELETE_CHECK

#include <iostream>
#include "cpp/source_files/CompressedFileTrain.h"
#include "cpp/source_files/SoftChunk.h"

int main() {
    using namespace std;
    using namespace iman;

    CompressedFileTrain train("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0A01z", true);
    train.open();

    cout << train << endl;
    for (auto pfile: train){
        cout << *pfile << endl;
        cout << pfile->getSoftChunk() << endl;
    }

    return 0;
}
