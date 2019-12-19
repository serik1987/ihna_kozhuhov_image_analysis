//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_HARDCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_HARDCHUNK_H

#include "Chunk.h"

namespace ihna::kozhukhov::image_analysis {

    /**
     * Defines properties of an experimental setup
     */
    class HardChunk: public Chunk {
    public:
#pragma PACK(push, 1)
        struct HARD_CHUNK{
            char		    Tag[4]              ="\x00\x00\x00";//01 01
            char		    CameraName[16]      ="\x00";        //02 05
            uint32_t		CameraType          =0;             //03 06
            uint32_t		ResolutionX         =0;             //04 07 Physical X resolution of CCD chip
            uint32_t		ResolutionY         =0;             //05 08 Physical Y resolution of CCD chip
            uint32_t		PixelSizeX          =0;             //06 09 micrometers
            uint32_t		PixelSizeY          =0;             //07 10 micrometers
            uint32_t		CCDApertureX        =0;             //08 11 micrometers
            uint32_t		CCDApertureY        =0;             //09 12 micrometers
            uint32_t		IntegrationTime     =0;             //10 13 microseconds
            uint32_t		InterFrameTime      =0;             //11 14 microseconds
            uint32_t		HardwareBinningV    =0;             //12 15 Vertical   or X binning
            uint32_t		HardwareBinningH    =0;             //13 16 Horizontal or Y binning
            uint32_t		HardwareGain        =0;             //14 17
            int32_t		    HardwareOffset      =0;             //15 18
            uint32_t		CCDSizeX            =0;             //16 19 Hardware binned CCD X size
            uint32_t		CCDSizeY            =0;             //17 20 Hardware binned CCD Y size
            uint32_t		DynamicRange        =0;             //18 21
            uint32_t		OpticsFocalLengthTop=0;             //19 22 Top lens focal lenght (the one closer
                                                                // to the camera), millimeters
            uint32_t		OpticsFocalLengthBottom=0;          //20 23 Bottom lens focal lenght, millimeters
            uint32_t        HardwareBits        =0;             //21 24 Hardware digitization, bits
            char            Comments[160]       ="";            //22 64
        };
#pragma PACK(pop)

    private:
        HARD_CHUNK info;

    public:
        explicit HardChunk(uint32_t size): Chunk("HARD", size) {
            body = (char*)&info;
        };

        /**
         *
         * @return chunk tag
         */
        [[nodiscard]] const char* getTag() const { return info.Tag; }

        /**
         *
         * @return Camera name
         */
        [[nodiscard]] std::string getCameraName() const { return readString(info.CameraName, 16); }

        /**
         *
         * @return camera type
         */
        [[nodiscard]] uint32_t getCameraType() const { return info.CameraType; }

        /**
         *
         * @return Resolution on X
         */
        [[nodiscard]] uint32_t getResolutionX() const { return info.ResolutionX; }

        /**
         *
         * @return Resolution on Y
         */
        [[nodiscard]] uint32_t getResolutionY() const { return info.ResolutionY; }

        /**
         *
         * @return pixel size on X in um
         */
        [[nodiscard]] uint32_t getPixelSizeX() const { return info.PixelSizeX; }

        /**
         *
         * @return pixel size on Y in um
         */
        [[nodiscard]] uint32_t getPixelSizeY() const { return info.PixelSizeY; }

        /**
         *
         * @return CCD aperture on X in um
         */
        [[nodiscard]] uint32_t getCcdApertureX() const { return info.CCDApertureX; }

        /**
         *
         * @return CCD aperture on Y in um
         */
        [[nodiscard]] uint32_t getCcdApertureY() const { return info.CCDApertureY; }

        /**
         *
         * @return integration time in usec
         */
        [[nodiscard]] uint32_t getIntegrationTime() const { return info.IntegrationTime; }

        /**
         *
         * @return interframe time in usec
         */
        [[nodiscard]] uint32_t getInterframeTime() const { return info.InterFrameTime; }

        /**
         *
         * @return Vertical   or X binning
         */
        [[nodiscard]] uint32_t getVerticalHardwareBinning() const { return info.HardwareBinningV; }

        /**
         *
         * @return Horizontal or Y binning
         */
        [[nodiscard]] uint32_t getHorirontalHardwareBinning() const { return info.HardwareBinningH; }

        /**
         *
         * @return hardware gain
         */
        [[nodiscard]] uint32_t getHardwareGain() const { return info.HardwareGain; }

        /**
         *
         * @return hardware offset
         */
        [[nodiscard]] int32_t getHardwareOffset() const { return info.HardwareOffset; }

        /**
         *
         * @return Hardware binned CCD X size
         */
        [[nodiscard]] uint32_t getCcdSizeX() const { return info.CCDSizeX; }

        /**
         *
         * @return Hardware binned CCD Y size
         */
        [[nodiscard]] uint32_t getCcdSizeY() const { return info.CCDSizeY; }

        /**
         *
         * @return dynamic range
         */
        [[nodiscard]] uint32_t getDynamicRange() const { return info.DynamicRange; }

        /**
         *
         * @return Top lens focal length (the one closer to the camera), millimeters
         */
        [[nodiscard]] uint32_t getOpticsFocalLengthTop() const { return info.OpticsFocalLengthTop; }

        /**
         *
         * @return Bottom lens focal length, millimeters
         */
        [[nodiscard]] uint32_t getOpticsFocalLengthBottom() const { return info.OpticsFocalLengthBottom; }

        /**
         *
         * @return Hardware bits
         */
        [[nodiscard]] uint32_t getHardwareBits() const { return info.HardwareBits; }

        friend std::ostream& operator<<(std::ostream& out, const HardChunk& chunk);
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_HARDCHUNK_H
