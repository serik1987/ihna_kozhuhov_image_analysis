//
// Created by serik1987 on 20.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_INIT_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_INIT_H

#define GLOBAL_NAMESPACE ihna::kozhukhov::image_analysis

#define MSG_NOT_ANALYSIS_FILE_EXCEPTION "The file is not an analysis source file"
#define MSG_COMP_CHUNK_NOT_EXIST_EXCEPTION "The following file doesn't contain COMP chunk"
#define MSG_NOT_COMPRESSED_SOURCE_FILE "This is not a compressed source file"
#define MSG_DATA_CHUNK_NOT_READ_EXCEPTION "The readFromFile function allows you to read the data from file header or file footer."\
" Its function is not to read the data from the file body. Since the DATA chunk "\
"corresponds to the file body, call DataChunk::readFromFile is considered to be an "\
"error itself"
#define MSG_EXPERIMENT_MODE_EXCEPTION "The function is not applicable for this stimulation mode"
#define MSG_SYNCHRONIZATION_CHANNEL_NUMBER_EXCEPTION "The number of synchronization channel passed is out of range"
#define MSG_UNSUPPORTED_EXPERIMENT_MODE_EXCEPTION "The stimulation protocol is unknown or unsupported by this version of the program"
#define MSG_FRAME_NUMBER_MISMATCH "Total number of frames written in ISOI chunk is no the same as " \
"total number of frames found in the whole record"
#define MSG_DATA_CHUNK_SIZE_MISMATCH "The DATA chunk size is not enough to encompass all frames"
#define MSG_ISOI_CHUNK_SIZE_MISMATCH "The ISOI chunk size is not enough to encompass all chunks"
#define MSG_FILE_SIZE_MISMATCH "The file size is not enough to encompass the ISOI chunk"
#define MSG_EXPERIMENTAL_CHUNK_NOT_FOUND "Necessary experimental chunk (COST for continuous stimulation,"\
"EPST for episodic stimulation) is absent in the following file"
#define MSG_FILE_HEADER_MISMATCH "Size of the file header is not the same as header size of the " \
"head file"
#define MSG_FRAME_HEADER_MISMATCH "Frame header for the file is not the same as frame header for " \
"the head file"
#define MSG_MAP_DIMENSIONS_MISMATCH "Frame resolution for this file is not the same as frame " \
"resolution for the head file"
#define MSG_DATA_TYPE_MISMATCH "Data type for this file is not the same as data type for " \
"the head file"
#define MSG_GREEN_FILE_EXCEPTION "The file is not a green source file"
#define MSG_FILE_OPEN_EXCEPTION "Error in opening the file"
#define MSG_FILE_READ_EXCEPTION "Error in reading the file"
#define MSG_UNSUPPORTED_CHUNK_EXCEPTION "Chunk '" + std::string(id, CHUNK_ID_SIZE) + \
"' is presented in the file but not supported by the current version of image-analysis"
#define MSG_CHUNK_SIZE_MISMATCH_EXCEPTION "Chunk '" + std::string(id, CHUNK_ID_SIZE) + \
"' has an actual size that is much different than the desired size"
#define MSG_CHUNK_NOT_FOUND_EXCEPTION "Chunk '" + name + "' was not found in the source file"
#define MSG_FILE_NOT_OPENED_EXCEPTION operation + " method was applied before the file opening for"
#define MSG_FILE_NOT_ISOI_EXCEPTION "The file doesn't relate to the IMAN source file because of errors in ISOI chunk"
#define MSG_FILE_NOT_LOADED_EXCEPTION "The method " + methodName + " can't be applied until the file info will be loaded" \
" by means of loadFileInfo"
#define MSG_DATA_CHUNK_NOT_FOUND_EXCEPTION "No DATA chunk was found in the file"
#define MSG_NOT_STREAM_FILE "This file is not a stream source file"
#define MSG_NOT_TRAIN_HEAD "The file is not the first file in the train"

#define MSG_FRAME_NOT_READ "The function can't be executed and return the result because the frame has not been "\
"loaded from the hard disk"
#define MSG_FRAME_OUT_OF_RANGE "Trying to read/seek the frame that is out of range"
#define MSG_COMPRESSED_FRAME_READ_ERROR "Only frame number 0 can be read from the compressed file train"
#define MSG_FRAM_CHUNK_NOT_FOUND "The frame reading is failed because the FRAM chunk is absent"

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_INIT_H
