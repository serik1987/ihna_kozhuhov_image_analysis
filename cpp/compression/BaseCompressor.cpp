//
// Created by serik1987 on 06.01.2020.
//

#include "BaseCompressor.h"
#include "../source_files/SoftChunk.h"

namespace GLOBAL_NAMESPACE{

    BaseCompressor::BaseCompressor(FileTrain &input, const std::string& output_path):
        train(input), full_output_train_file("") {
        this->output_path = output_path;
        original_frame_size = train.getFrameSize();
        elements_in_frame = original_frame_size * sizeof(CompressedElement) / sizeof(OriginalElement);

        try{
            previous_frame = new OriginalElement[elements_in_frame];
            current_frame = new OriginalElement[elements_in_frame];
            compressed_frame = new CompressedElement[elements_in_frame];
            extra_pixels = new EXTRA_PIXEL_RECORD[elements_in_frame];
        } catch (std::exception& e){
            delete [] previous_frame;
            delete [] current_frame;
            delete [] compressed_frame;
            delete [] extra_pixels;
        }
    }

    BaseCompressor::~BaseCompressor() {
        delete [] previous_frame;
        delete [] current_frame;
        delete [] compressed_frame;
        delete [] extra_pixels;
    }

    std::ostream &operator<<(std::ostream &out, const BaseCompressor &compressor) {
        using namespace std;

        out << "Source file: " << compressor.train.getFilename() << endl;
        out << "Original frame size: " << compressor.original_frame_size << endl;
        out << "Compressed frame size: " << compressor.elements_in_frame << endl;

        return out;
    }

    void BaseCompressor::run() {
        using namespace std;

        int idx = 0;
        for (auto* file: train){
            const string& input_dir(file->getFilePath());
            const string& input_file(file->getFileName());
            const string& output_dir(output_path);
            string output_file = getOutputFile(input_file);
            full_input_file = input_dir + input_file;
            full_output_file = output_dir + output_file;
            if (full_output_train_file.empty()){
                full_output_train_file = full_output_file;
            }
            compressed_frame_number = file->getSoftChunk().getFramesThisFile();

            ofstream output;
            output.open(output_dir + output_file, ios_base::out | ios_base::trunc | ios_base::binary);
            if (output.fail()){
                throw SourceFile::file_open_exception(output_dir + output_file, train.getFilename());
            }

            writeHeader(*file, output);

            writeFirstFrame(file->getFileStream(), output);
            for (int n = 1; n < file->getSoftChunk().getFramesThisFile(); n++){
                writeConsequtiveFrame(file->getFileStream(), output);
            }

            output.close();

            idx++;
            if (progressFunction != nullptr){
                progressFunction((float)idx * 100 / train.getFileNumber(), handle);
            }
        }
    }

    void BaseCompressor::writeFirstFrame(std::istream &in, std::ostream &out) {
        in.read((char*)previous_frame, original_frame_size);
        if (in.fail()){
            throw SourceFile::file_read_exception(full_input_file);
        }
        out.write((char*)previous_frame, original_frame_size);
        if (out.fail()){
            throw SourceFile::file_write_exception(full_output_file);
        }
    }
}