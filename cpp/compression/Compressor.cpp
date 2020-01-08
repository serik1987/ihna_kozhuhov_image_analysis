//
// Created by serik1987 on 06.01.2020.
//

#include "Compressor.h"
#include "../source_files/IsoiChunk.h"
#include "../source_files/SoftChunk.h"
#include "../source_files/CompChunk.h"

namespace GLOBAL_NAMESPACE{

    std::string Compressor::getOutputFile(const std::string &input_file) {
        if (input_file.length() >= 16){
            return input_file;
        } else {
            return input_file + "z";
        }
    }

    void Compressor::writeHeader(TrainSourceFile &input, std::ofstream &output) {
        auto& isoi = input.getIsoiChunk();

        isoi.write(full_output_file, output);
        for (auto& chunk: isoi){
            auto psoft = dynamic_cast<SoftChunk*>(&chunk);
            if (psoft != nullptr){
                writeSoftChunk(psoft, output);
                writeCompChunk(output);
            } else {
                chunk.write(full_output_file, output);
            }
        }
    }

    void Compressor::writeSoftChunk(const SoftChunk* psoft, std::ofstream &output) {
        SoftChunk new_soft(*psoft);
        if (!new_soft.getCurrentFilename().empty()) {
            new_soft.setCurrentFilename(getOutputFile(new_soft.getCurrentFilename()));
        }
        if (!new_soft.getNextFilename().empty()) {
            new_soft.setNextFilename(getOutputFile(new_soft.getNextFilename()));
        }
        if (!new_soft.getPreviousFilename().empty()) {
            new_soft.setPreviousFilename(getOutputFile(new_soft.getPreviousFilename()));
        }
        new_soft.setFileType(SourceFile::CompressedFile);
        new_soft.write(full_output_file, output);
    }

    void Compressor::writeCompChunk(std::ofstream &output) {
        CompChunk chunk(sizeof(EXTRA_PIXEL_RECORD), elements_in_frame, compressed_frame_number);
        chunk.write(full_output_file, output);
    }

    void Compressor::writeConsequtiveFrame(std::ifstream &in, std::ofstream &out) {
        extra_pixels_number = 0;
        in.read((char*)current_frame, original_frame_size);
        if (in.fail()){
            throw SourceFile::file_read_exception(full_input_file);
        }

        for (int i=0; i < elements_in_frame; ++i) {
            int diff = (int) current_frame[i] - (int) previous_frame[i];
            if (diff > 127 || diff < -128) {
                int idx = extra_pixels_number++;
                COMPRESSOR_RECORD rec;
                rec.L = current_frame[i];
                compressed_frame[i] = rec.S[0];
                extra_pixels[idx].L = i;
                extra_pixels[idx].S[3] = rec.S[1];
            } else {
                compressed_frame[i] = (int8_t) diff;
            }
        }

        out.write((char*)compressed_frame, elements_in_frame);
        out.write((char*)&extra_pixels_number, sizeof(extra_pixels_number));
        if (extra_pixels_number > 0){
            out.write((char*)extra_pixels, extra_pixels_number * sizeof(EXTRA_PIXEL_RECORD));
        }
        if (out.fail()){
            throw SourceFile::file_write_exception(full_output_file);
        }

        auto* temp_frame = current_frame;
        current_frame = previous_frame;
        previous_frame = temp_frame;
    }
}