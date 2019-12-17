//
// Created by serik1987 on 16.12.2019.
//

#include "SoftChunk.h"

namespace iman{

    std::ostream &operator<<(std::ostream &out, const SoftChunk &chunk) {
        using std::endl;

        out << "===== SOFT =====\n";
        out << "Tag: " << std::string(chunk.getTag(), 4) << "\n";
        out << "Record date and time: " << std::string(chunk.getDateTimeRecorded()) << "\n";
        out << "User name: " << chunk.getUserName() << endl;
        out << "Subject ID: " << chunk.getSubjectId() << endl;
        out << "Current filename: " << chunk.getCurrentFilename() << endl;
        out << "Previous filename: " << chunk.getPreviousFilename() << endl;
        out << "Next filename: " << chunk.getNextFilename() << endl;
        out << "Data type for a single pixel: " << chunk.getDataType() << endl;
        out << "File type: " << chunk.getFileType() << endl;
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
}