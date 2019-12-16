//
// Created by serik1987 on 16.12.2019.
//

#ifndef IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOFTCHUNK_H
#define IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOFTCHUNK_H

#include "Chunk.h"

namespace iman {

    /**
     * Defines general properties of the recorded signal.
     * This is the main file that shall be presented in any source file
     * The files will be failed to be open without this chunk
     */
    class SoftChunk: public Chunk {
    public:
#pragma PACK(push, 1)
        struct SOFT_CHUNK{
            char		Tag[4] = "\x00\x00\x00";			//01 01
            char		DateTimeRecorded[24] = "";	//02 07 ASCII UNIX time
            char		UserName[16] = "";		//03 11 ASCII
            char		SubjectID[16] = "";		//04 15 ASCII
            char		ThisFilename[16] = "";	//05 19 Encoded in MakeFileName
            char		PrevFilename[16] = "";	//06 23
            char		NextFilename[16] = "";	//07 27
            uint32_t		DataType = 0;		//08 28
            uint32_t		FileType = 0;		//09 29
            uint32_t		SizeOfDataType = 0;		//10 30
            uint32_t		XSize = 0;			//11 31 X size of stored frames
            uint32_t		YSize = 0;			//12 32 Y size of stored frames
            uint32_t		ROIXPosition = 0;		//13 33 X coordinate of upper-left conner of ROI (before binning)
            uint32_t		ROIYPosition = 0;		//14 34 Y coordinate of upper-left conner of ROI (before binning)
            uint32_t		ROIXSize = 0;		//15 35 X size of ROI (before binning)
            uint32_t		ROIYSize = 0;		//16 36 Y size of ROI (before binning)
            uint32_t		ROIXPositionAdjusted = 0;	//17 37 X coordinate of upper-left conner of adjusted ROI (before binning)
            uint32_t		ROIYPositionAdjusted = 0;	//18 38 Y coordinate of upper-left conner of adjusted ROI (before binning)
            uint32_t		ROINumber = 0;		//19 39 Sequential ROI number. Set to 0 for main ROI
            uint32_t		TemporalBinning = 0;	//20 40
            uint32_t		SpatialBinningX = 0;	//21 41 X
            uint32_t		SpatialBinningY = 0;	//22 42 Y
            uint32_t		FrameHeaderSize = 0;	//23 43 Size of the header for each frame
            uint32_t		NFramesTotal = 0;		//24 44 Expected number of frames
            uint32_t		NFramesThisFile = 0;	//25 45 Number of frames in this file
            uint32_t		WaveLength = 0;		//26 46 Wave length, nanometers
            uint32_t		FilterWidth = 0;		//27 47 Filter width, nanometers
            char		Comments[68] = "";		//28 64
        };
#pragma PACK(pop)

    private:
        SOFT_CHUNK info;

    public:
        /**
         * Initializes the chunk
         *
         * @param size size of chunk body
         */
        explicit SoftChunk(uint32_t size): Chunk("SOFT", size) {
            body = (char*)&info;
        }

        /**
         *
         * @return chunk tag
         */
        [[nodiscard]] const char* getTag() const { return info.Tag; }

        /**
         *
         * @return Actual date and time where the data were recorded
         */
        [[nodiscard]] std::string getDateTimeRecorded() const { return readString(info.DateTimeRecorded, 24); }

        /**
         *
         * @return User name
         */
        [[nodiscard]] std::string getUserName() const { return readString(info.UserName, 16); }

        /**
         *
         * @return subject ID
         */
        [[nodiscard]] std::string getSubjectId() const { return readString(info.SubjectID, 16); }

        /**
         *
         * @return current filename
         */
        [[nodiscard]] std::string getCurrentFilename() const { return readString(info.ThisFilename, 16); }

        /**
         *
         * @return next filename or empty string if this is the last filename
         */
        [[nodiscard]] std::string getNextFilename() const { return readString(info.NextFilename, 16); }

        /**
         *
         * @return previous filename or empty string if this is previous filename
         */
        [[nodiscard]] std::string getPreviousFilename()const { return readString(info.PrevFilename, 16); }

        /**
         *
         * @return space that a single map's pixel stores on the disk
         */
        [[nodiscard]] uint32_t getDataType() const { return info.DataType; }

        /**
         *
         * @return file type
         */
        [[nodiscard]] uint32_t getFileType() const { return info.FileType; }

        /**
         *
         * @return space that a single map's pixel stores on the disk
         */
        [[nodiscard]] uint32_t getDataTypeSize() const { return info.SizeOfDataType; }

        /**
         *
         * @return map's X size in pixels
         */
        [[nodiscard]] uint32_t getXSize() const { return info.XSize; }

        /**
         *
         * @return map's Y size in pixels
         */
        [[nodiscard]] uint32_t getYSize() const { return info.YSize; }

        /**
         *
         * @return X coordinate of upper-left conner of ROI (before binning)
         */
        [[nodiscard]] uint32_t getRoiXPosition() const { return info.ROIXPosition; }

        /**
         *
         * @return Y coordinate of upper-left conner of ROI (before binning)
         */
        [[nodiscard]] uint32_t getRoiYPosition() const { return info.ROIYPosition; }

        /**
         *
         * @return X size of ROI (before binning)
         */
        [[nodiscard]] uint32_t getRoiXSize() const { return info.ROIXSize; }

        /**
         *
         * @return Y size of ROI (before binning)
         */
        [[nodiscard]] uint32_t getRoiYSize() const { return info.ROIYSize; }

        /**
         *
         * @return X coordinate of upper-left conner of adjusted ROI (before binning)
         */
        [[nodiscard]] uint32_t getRoiXPositionAdjusted() const { return info.ROIXPositionAdjusted; }

        /**
         *
         * @return Y coordinate of upper-left conner of adjusted ROI (before binning)
         */
        [[nodiscard]] uint32_t getRoiYPositionAdjusted() const { return info.ROIYPositionAdjusted; }

        /**
         *
         * @return Sequential ROI number. Set to 0 for main ROI
         */
        [[nodiscard]] uint32_t getRoiNumber() const { return info.ROINumber; }

        /**
         *
         * @return Temporal binning
         */
        [[nodiscard]] uint32_t getTemporalBinning() const { return info.TemporalBinning; }

        /**
         *
         * @return spatial binning on X
         */
        [[nodiscard]] uint32_t getSpatialBinningX() const { return info.SpatialBinningX; }

        /**
         *
         * @return spatial binning on Y
         */
        [[nodiscard]] uint32_t getSpatialBinningY() const { return info.SpatialBinningY; }

        /**
         *
         * @return frame header size, bytes
         */
        [[nodiscard]] uint32_t getFrameHeaderSize() const { return info.FrameHeaderSize; }

        /**
         *
         * @return total frames in the record
         */
        [[nodiscard]] uint32_t getTotalFrames() const { return info.NFramesTotal; }

        /**
         *
         * @return number of frames in this file
         */
        [[nodiscard]] uint32_t getFramesThisFile() const { return info.NFramesThisFile; }

        /**
         *
         * @return wavelength in nm
         */
        [[nodiscard]] uint32_t getWavelengthNm() const { return info.WaveLength; }

        /**
         *
         * @return filter width in nm
         */
        [[nodiscard]] uint32_t getFilterWidth() const { return info.FilterWidth; }
    };

}


#endif //IHNA_KOZHUHOV_IMAGE_ANALYSIS_SOFTCHUNK_H
