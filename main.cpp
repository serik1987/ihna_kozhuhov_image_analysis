/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.image_analysis
 */

#include <iostream>
#include "cpp/source_files/CompressedSourceFile.h"
#include "cpp/source_files/SoftChunk.h"

int main() {
    using namespace std;
    using namespace iman;

    CompressedSourceFile sourceFile("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0A01z",
            TrainSourceFile::NotInHeadTraverse);
    sourceFile.open();
    sourceFile.loadFileInfo();

    cout << sourceFile << endl;
    cout << sourceFile.getSoftChunk() << endl;
    return 0;
}
