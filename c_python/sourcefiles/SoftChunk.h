//
// Created by serik1987 on 24.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SOFTCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SOFTCHUNK_H

#include "../../cpp/source_files/SoftChunk.h"


extern "C" {

    static int PyImanS_SoftChunk_Init(PyImanS_SoftChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "SOFT", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_SoftChunk_GetProperty(PyImanS_SoftChunkObject* self, PyObject* key){
        using namespace GLOBAL_NAMESPACE;
        PyObject* result;
        auto* chunk = (SoftChunk*)self->super.handle;

        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_KeyError, "The chunk properties are indexed by their names");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);

        if (result != NULL) {
            return result;
        } else if (strcmp(name, "file_type") == 0) {
            SourceFile::FileType type = chunk->getFileType();
            if (type == SourceFile::AnalysisFile) {
                return PyUnicode_FromString("analysis");
            }
            if (type == SourceFile::GreenFile) {
                return PyUnicode_FromString("green");
            }
            if (type == SourceFile::StreamFile) {
                return PyUnicode_FromString("stream");
            }
            if (type == SourceFile::CompressedFile) {
                return PyUnicode_FromString("compressed");
            }
            return PyUnicode_FromString("unknown");
        } else if (strcmp(name, "date_time_recorded") == 0) {
            return PyUnicode_FromString(chunk->getDateTimeRecorded().c_str());
        } else if (strcmp(name, "user_name") == 0) {
            return PyUnicode_FromString(chunk->getUserName().c_str());
        } else if (strcmp(name, "subject_id") == 0) {
            return PyUnicode_FromString(chunk->getSubjectId().c_str());
        } else if (strcmp(name, "current_filename") == 0) {
            return PyUnicode_FromString(chunk->getCurrentFilename().c_str());
        } else if (strcmp(name, "previous_filename") == 0) {
            return PyUnicode_FromString(chunk->getPreviousFilename().c_str());
        } else if (strcmp(name, "next_filename") == 0) {
            return PyUnicode_FromString(chunk->getNextFilename().c_str());
        } else if (strcmp(name, "data_type") == 0) {
            return PyLong_FromUnsignedLong(chunk->getDataType());
        } else if (strcmp(name, "pixel_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getDataTypeSize());
        } else if (strcmp(name, "x_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getXSize());
        } else if (strcmp(name, "y_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getYSize());
        } else if (strcmp(name, "roi_x_position") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRoiXPosition());
        } else if (strcmp(name, "roi_y_position") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRoiYPosition());
        } else if (strcmp(name, "roi_x_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRoiXSize());
        } else if (strcmp(name, "roi_y_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRoiYSize());
        } else if (strcmp(name, "roi_x_position_adjusted") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRoiXPositionAdjusted());
        } else if (strcmp(name, "roi_y_position_adjusted") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRoiYPositionAdjusted());
        } else if (strcmp(name, "roi_number") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRoiNumber());
        } else if (strcmp(name, "temporal_binning") == 0) {
            return PyLong_FromUnsignedLong(chunk->getTemporalBinning());
        } else if (strcmp(name, "spatial_binning_x") == 0) {
            return PyLong_FromUnsignedLong(chunk->getSpatialBinningX());
        } else if (strcmp(name, "spatial_binning_y") == 0) {
            return PyLong_FromUnsignedLong(chunk->getSpatialBinningY());
        } else if (strcmp(name, "frame_header_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getFrameHeaderSize());
        } else if (strcmp(name, "total_frames") == 0) {
            return PyLong_FromUnsignedLong(chunk->getTotalFrames());
        } else if (strcmp(name, "frames_this_file") == 0) {
            return PyLong_FromUnsignedLong(chunk->getFramesThisFile());
        } else if (strcmp(name, "wavelength") == 0) {
            return PyLong_FromUnsignedLong(chunk->getWavelengthNm());
        } else if (strcmp(name, "filter_width") == 0) {
            return PyLong_FromUnsignedLong(chunk->getFilterWidth());
        } else {
            PyErr_SetString(PyExc_KeyError, "name of the SOFT chunk property is not correct");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_SoftChunk_mapping = {
        NULL,
        (binaryfunc)PyImanS_SoftChunk_GetProperty,
        NULL,
    };

    static int PyImanS_SoftChunk_Create(PyObject* module){

        PyImanS_SoftChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_SoftChunkType.tp_doc = "Use SoftChunk instead";
        PyImanS_SoftChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_SoftChunkType.tp_init = (initproc)PyImanS_SoftChunk_Init;
        PyImanS_SoftChunkType.tp_as_mapping = &PyImanS_SoftChunk_mapping;

        if (PyType_Ready(&PyImanS_SoftChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_SoftChunkType);

        if (PyModule_AddObject(module, "_sourcefiles_SoftChunk", (PyObject*)&PyImanS_SoftChunkType) < 0){
            Py_DECREF(&PyImanS_SoftChunkType);
            return -1;
        }

        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_SoftChunkType;

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SOFTCHUNK_H
