/**
 * This is an auxiliary file that has been created to test feasibilities of the ihna.kozhukhov.imageanalysis
 */

#include "cpp/source_files/StreamFileTrain.h"
#include "cpp/isolines/TimeAverageIsoline.h"

#define WORKING_DIR "/home/serik1987/vasomotor-oscillations/c022/"

using namespace ihna::kozhukhov::image_analysis;

bool progress_function(int completed, int total, const char* msg, void* handle){
    std::cout << msg << std::endl;
    std::cout << handle << std::endl;
    std::cout << (double)completed * 100 / total << "% completed\n";
}


int main() {
    using namespace std;

    {



    }
    return 0;
}
