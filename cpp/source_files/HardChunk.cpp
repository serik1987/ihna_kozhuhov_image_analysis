//
// Created by serik1987 on 16.12.2019.
//

#include <bitset>
#include "HardChunk.h"

namespace ihna::kozhukhov::image_analysis{

    std::ostream &operator<<(std::ostream &out, const HardChunk &chunk) {
        out << "===== HARD =====\n";
        out << "Camera name: " << chunk.getCameraName() << "\n";
        out << "Camera type: " << chunk.getCameraType() << "\n";
        out << "Physical resolution of CCD chip, X: " << chunk.getResolutionX() << "\n";
        out << "Physical resolution of CCD chip, Y: " << chunk.getResolutionY() << "\n";
        out << "Approximate size of one pixel, um, X: " << chunk.getPixelSizeX() << "\n";
        out << "Approximate size of one pixel, um, Y: " << chunk.getPixelSizeY() << "\n";
        out << "CCD aperture on X, um: " << chunk.getCcdApertureX() << "\n";
        out << "CCD aperture on Y, um: " << chunk.getCcdApertureY() << "\n";
        out << "Integration time, usec: " << chunk.getIntegrationTime() << "\n";
        out << "Interframe time, usec: " << chunk.getInterframeTime() << "\n";
        out << "Vertical hardware binning: " << chunk.getVerticalHardwareBinning() << "\n";
        out << "Horizontal hardware binning: " << chunk.getHorirontalHardwareBinning() << "\n";
        out << "Hardware gain: " << chunk.getHardwareGain() << "\n";
        out << "Hardware offset: " << chunk.getHardwareOffset() << "\n";
        out << "Hardware binned CCD X size: " << chunk.getCcdSizeX() << "\n";
        out << "Hardware binned CCD Y size: " << chunk.getCcdSizeY() << "\n";
        out << "Dynamic range: " << chunk.getDynamicRange() << "\n";
        out << "Top lens focal length (the one closer to the camera), millimeters: "
            << chunk.getOpticsFocalLengthTop() << "\n";
        out << "Bottom lens focal length, millimeters: " << chunk.getOpticsFocalLengthBottom() << "\n";
        std::bitset<32> bs(chunk.getHardwareBits());
        out << "Hardware bits: " << bs;

        return out;
    }
}