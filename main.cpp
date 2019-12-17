/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.image_analysis
 */

#include <iostream>
#include "cpp/source_files/SourceFile.h"
#include "cpp/source_files/Chunk.h"

int main() {
    using namespace std;
    using namespace iman;

    SourceFile sourceFile("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0A01z");
    sourceFile.open();
    auto mainChunkHeader = sourceFile.readChunkHeader(true);
    if (mainChunkHeader != "ISOI"){
        throw std::runtime_error("Old file format is still not supported");
    }
    auto* softChunk = sourceFile.findChunk("SOFT");

    cout << "File path: " << sourceFile.getFilePath() << endl;
    cout << "File name: " << sourceFile.getFileName() << endl;
    cout << "Full file path: " << sourceFile.getFullname() << endl;
    if (sourceFile.isOpened()){
        cout << "File opened\n";
    }
    cout << *softChunk << endl;


    delete softChunk;
    return 0;
}
