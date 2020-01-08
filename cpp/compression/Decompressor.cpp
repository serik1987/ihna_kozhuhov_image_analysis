//
// Created by serik1987 on 06.01.2020.
//

#include "Decompressor.h"
#include "../source_files/IsoiChunk.h"
#include "../source_files/SoftChunk.h"

namespace GLOBAL_NAMESPACE{

    std::string Decompressor::getOutputFile(const std::string &input_file) {
        auto S = input_file.length();
        if (input_file[S-1] == 'z'){
            return input_file.substr(0, S-1);
        } else {
            return input_file;
        }
    }

    void Decompressor::writeHeader(TrainSourceFile &input, std::ofstream &output) {
        auto& isoi = input.getIsoiChunk();
        isoi.write(full_output_file, output);

        for (auto& chunk: isoi){
            auto* psoft = dynamic_cast<SoftChunk*>(&chunk);
            auto* pcomp = dynamic_cast<CompChunk*>(&chunk);

            if (psoft != nullptr){
                writeSoftChunk(*psoft, output);
            } else if (pcomp != nullptr){
                checkCompChunk(*pcomp);
            } else {
                chunk.write(full_output_file, output);
            }
        }
    }

    void Decompressor::writeSoftChunk(const SoftChunk &input_chunk, std::ofstream& output) {
        SoftChunk output_chunk = input_chunk;
        output_chunk.setFileType(SourceFile::StreamFile);
        if (!input_chunk.getCurrentFilename().empty()){
            output_chunk.setCurrentFilename(getOutputFile(input_chunk.getCurrentFilename()));
        }
        if (!input_chunk.getNextFilename().empty()){
            output_chunk.setNextFilename(getOutputFile(output_chunk.getNextFilename()));
        }
        if (!input_chunk.getPreviousFilename().empty()){
            output_chunk.setPreviousFilename(getOutputFile(output_chunk.getPreviousFilename()));
        }
        output_chunk.write(full_output_file, output);
    }

    void Decompressor::checkCompChunk(const CompChunk &input_chunk) {
        if (input_chunk.getCompressedRecordSize() != sizeof(EXTRA_PIXEL_RECORD)){
            throw decompression_exception();
        }
        if (input_chunk.getCompressedFrameNumber() != compressed_frame_number){
            throw decompression_exception();
        }
        if (input_chunk.getCompressedFrameSize() != elements_in_frame){
            throw decompression_exception();
        }
    }

    void Decompressor::writeConsequtiveFrame(std::ifstream &in, std::ofstream &out) {
        in.read((char*)compressed_frame, elements_in_frame);
        if (in.fail()){
            throw SourceFile::file_read_exception(full_input_file);
        }

        for (int i=0; i < elements_in_frame; ++i){
            current_frame[i] = (uint16_t)((int)previous_frame[i] + (int8_t)compressed_frame[i]);
        }

        in.read((char*)&extra_pixels_number, sizeof(extra_pixels_number));
        in.read((char*)extra_pixels, extra_pixels_number * sizeof(EXTRA_PIXEL_RECORD));

        for (int i=0; i < extra_pixels_number; ++i){
            COMPRESSOR_RECORD rec;

            rec.S[1] = extra_pixels[i].S[3];
            extra_pixels[i].S[3] = 0;
            int idx = extra_pixels[i].L;
            rec.S[0] = compressed_frame[idx];
            current_frame[idx] = rec.L;
        }

        out.write((char*)current_frame, original_frame_size);
        if (out.fail()){
            throw SourceFile::file_write_exception(full_output_file);
        }

        auto* temp_frame = current_frame;
        current_frame = previous_frame;
        previous_frame = temp_frame;
    }
}