/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.image_analysis
 */

#include <iostream>
#include "cpp/source_files/SourceFile.h"

int main() {
    using namespace std;
    using namespace iman;

    SourceFile sourceFile("/home/serik1987/vasomotor-oscillations/sample_data/c022z/", "T_1BF.0A00z");
    sourceFile.open();
    auto mainChunkHeader = sourceFile.readChunkHeader(true);
    if (mainChunkHeader != "ISOI"){
        throw std::runtime_error("Old file format is still not supported");
    }
    auto softChunkHeader = sourceFile.findChunkHeader("SOFT");

    cout << "File path: " << sourceFile.getFilePath() << endl;
    cout << "File name: " << sourceFile.getFileName() << endl;
    cout << "Full file path: " << sourceFile.getFullname() << endl;
    if (sourceFile.isOpened()){
        cout << "File opened\n";
    }
    cout << "Main chunk ID: " << mainChunkHeader.getChunkId() << endl;
    cout << "Main chunk size: " << mainChunkHeader.getChunkSize() << endl;
    cout << "ISOI chunk header ID: " << softChunkHeader.getChunkId() << endl;
    cout << "ISOI chunk header size: " << softChunkHeader.getChunkSize() << endl;


    return 0;
}
