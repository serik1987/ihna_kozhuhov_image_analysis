//
// Created by serik1987 on 06.01.2020.
//

#include "BaseCompressor.h"

namespace GLOBAL_NAMESPACE{

    BaseCompressor::BaseCompressor(FileTrain &input, const std::string& output_path):
        train(input) {
        this->output_path = output_path;
        original_frame_size = train.getFrameSize();
        elements_in_frame = original_frame_size * sizeof(CompressedElement) / sizeof(OriginalElement);

        try{
            first_frame = new OriginalElement[elements_in_frame];
            current_frame = new OriginalElement[elements_in_frame];
            compressed_frame = new CompressedElement[elements_in_frame];
        } catch (std::exception& e){
            delete [] first_frame;
            delete [] current_frame;
            delete [] compressed_frame;
        }
    }

    BaseCompressor::~BaseCompressor() {
        delete [] first_frame;
        delete [] current_frame;
        delete [] compressed_frame;
    }

    std::ostream &operator<<(std::ostream &out, const BaseCompressor &compressor) {
        using namespace std;

        out << "Source file: " << compressor.train.getFilename() << endl;
        out << "Original frame size: " << compressor.original_frame_size << endl;
        out << "Compressed frame size: " << compressor.elements_in_frame << endl;

        return out;
    }

    void BaseCompressor::run() {
        using std::cout, std::endl;

        for (auto* file: train){
            const std::string& input_dir(file->getFilePath());
            const std::string& input_file(file->getFileName());
            const std::string& output_dir(output_path);
            std::string output_file = getOutputFile(input_file);

            cout << "Input path: " << input_dir << "\n";
            cout << "Input file: " << input_file << "\n";
            cout << "Output dir: " << output_dir << "\n";
            cout << "Output file: " << output_file << "\n";
            cout << "================================= \n";
        }
    }
}