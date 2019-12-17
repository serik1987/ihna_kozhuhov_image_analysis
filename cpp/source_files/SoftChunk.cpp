//
// Created by serik1987 on 16.12.2019.
//

#include "SoftChunk.h"

namespace iman{

    std::ostream &operator<<(std::ostream &out, const SoftChunk &chunk) {
        using std::endl;

        out << "===== SOFT =====\n";
        out << "Tag: " << std::string(chunk.getTag(), 4) << "\n";
        out << "File type: ";
        switch (chunk.getFileType()){
            case SourceFile::AnalysisFile:
                out << "Analysis file\n";
                break;
            case SourceFile::CompressedFile:
                out << "Compressed file\n";
                break;
            case SourceFile::GreenFile:
                out << "Green file\n";
                break;
            case SourceFile::StreamFile:
                out << "Stream file\n";
                break;
            case SourceFile::UnknownFile:
                out << "Unknown or unsupported file\n";
                break;
            default:
                out << "[Message for somebody who changed the module. "
                       "TO-DO: UPGRADE operator<<(std::steam, iman::SoftChunk)] IN SoftChunk.cpp";
                break;
        }
        out << "Record date and time: " << std::string(chunk.getDateTimeRecorded()) << "\n";
        out << "User name: " << chunk.getUserName() << endl;
        out << "Subject ID: " << chunk.getSubjectId() << endl;
        out << "Current filename: " << chunk.getCurrentFilename() << endl;
        out << "Previous filename: " << chunk.getPreviousFilename() << endl;
        out << "Next filename: " << chunk.getNextFilename() << endl;
        out << "Data type for a single pixel: " << chunk.getDataType() << endl;
        out << "File type: " << chunk.getFileSubtype() << endl;
        out << "Space stored by the single map pixel on the hard disk: " << chunk.getDataTypeSize() << endl;
        out << "Map resolution X: " << chunk.getXSize() << endl;
        out << "Map resolution Y: " << chunk.getYSize() << endl;
        out << "X coordinate of upper left corner for ROI: " << chunk.getRoiXPosition() << endl;
        out << "Y coordinate of upper left corner for ROI: " << chunk.getRoiYPosition() << endl;
        out << "X size of ROI, in pixels: " << chunk.getRoiXSize() << endl;
        out << "Y size of ROI, in pixels: " << chunk.getRoiYSize() << endl;
        out << "X coordinate of adjusted upper left corner for ROI: " << chunk.getRoiXPositionAdjusted() << endl;
        out << "Y coordinate of adjusted upper left corner for ROI: " << chunk.getRoiYPositionAdjusted() << endl;
        out << "ROI Number: " << chunk.getRoiNumber() << endl;
        out << "Temporal binning: " << chunk.getTemporalBinning() << endl;
        out << "Spatial binning X: " << chunk.getSpatialBinningX() << endl;
        out << "Spatial binning Y: " << chunk.getSpatialBinningY() << endl;
        out << "Frame header size, bytes: " << chunk.getFrameHeaderSize() << endl;
        out << "Total frames number: " << chunk.getTotalFrames() << endl;
        out << "Number of frames on this file: " << chunk.getFramesThisFile() << endl;
        out << "Wavelength, nm: " << chunk.getWavelengthNm() << endl;
        out << "Filter width, nm: " << chunk.getFilterWidth();

        return out;
    }

    SourceFile::FileType SoftChunk::getFileType() const {
        SourceFile::FileType type;

        switch (getTag()[0]){
            case 'A':
                type = SourceFile::AnalysisFile;
                break;
            case 'C':
                type = SourceFile::CompressedFile;
                break;
            case 'G':
            case 'E':
                type = SourceFile::GreenFile;
                break;
            case 'T':
                type = SourceFile::StreamFile;
                break;
            default:
                type = SourceFile::UnknownFile;
                break;
        }

        return type;
    }
}