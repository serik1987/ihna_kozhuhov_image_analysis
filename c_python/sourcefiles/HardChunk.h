//
// Created by serik1987 on 25.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_HARDCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_HARDCHUNK_H

#include "../../cpp/source_files/HardChunk.h"

extern "C" {

    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_HardChunkObject;

    static PyTypeObject PyImanS_HardChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._HardChunk",
            .tp_basicsize = sizeof(PyImanS_HardChunkObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_HardChunk_Init(PyImanS_HardChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "HARD", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_HardChunk_GetProperty(PyImanS_HardChunkObject* self, PyObject* key){
        using namespace GLOBAL_NAMESPACE;

        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "The subscript index for the HARD chunk must be a string containing "
                                             "a property name");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);
        auto* chunk = (HardChunk*)self->super.handle;

        if (result != NULL) {
            return result;
        } else if (strcmp(name, "camera_name") == 0) {
            return PyUnicode_FromString(chunk->getCameraName().c_str());
        } else if (strcmp(name, "camera_type") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCameraType());
        } else if (strcmp(name, "resolution_x") == 0) {
            return PyLong_FromUnsignedLong(chunk->getResolutionX());
        } else if (strcmp(name, "resolution_y") == 0) {
            return PyLong_FromUnsignedLong(chunk->getResolutionY());
        } else if (strcmp(name, "pixel_size_x") == 0) {
            return PyLong_FromUnsignedLong(chunk->getPixelSizeX());
        } else if (strcmp(name, "pixel_size_y") == 0) {
            return PyLong_FromUnsignedLong(chunk->getPixelSizeY());
        } else if (strcmp(name, "ccd_aperture_x") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCcdApertureX());
        } else if (strcmp(name, "ccd_aperture_y") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCcdApertureY());
        } else if (strcmp(name, "integration_time") == 0) {
            return PyLong_FromUnsignedLong(chunk->getIntegrationTime());
        } else if (strcmp(name, "interframe_time") == 0) {
            return PyLong_FromUnsignedLong(chunk->getInterframeTime());
        } else if (strcmp(name, "vertical_hardware_binning") == 0) {
            return PyLong_FromUnsignedLong(chunk->getVerticalHardwareBinning());
        } else if (strcmp(name, "horizontal_hardware_binning") == 0) {
            return PyLong_FromUnsignedLong(chunk->getHorirontalHardwareBinning());
        } else if (strcmp(name, "hardware_gain") == 0) {
            return PyLong_FromUnsignedLong(chunk->getHardwareGain());
        } else if (strcmp(name, "hardware_offset") == 0) {
            return PyLong_FromLong(chunk->getHardwareOffset());
        } else if (strcmp(name, "ccd_size_x") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCcdSizeX());
        } else if (strcmp(name, "ccd_size_y") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCcdSizeY());
        } else if (strcmp(name, "dynamic_range") == 0) {
            return PyLong_FromUnsignedLong(chunk->getDynamicRange());
        } else if (strcmp(name, "optics_focal_length_top") == 0) {
            return PyLong_FromUnsignedLong(chunk->getOpticsFocalLengthTop());
        } else if (strcmp(name, "optics_focal_length_bottom") == 0) {
            return PyLong_FromUnsignedLong(chunk->getOpticsFocalLengthBottom());
        } else if (strcmp(name, "hardware_bits") == 0){
            return PyLong_FromUnsignedLong(chunk->getHardwareBits());
        } else {
            PyErr_SetString(PyExc_IndexError, "The subscript index doesn't refer to the valid property of the "
                                              "HARD chunk");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_HardChunkMapping = {
            .mp_length = NULL,
            .mp_subscript = (binaryfunc)PyImanS_HardChunk_GetProperty,
            .mp_ass_subscript = NULL,
    };

    static int PyImanS_HardChunk_Create(PyObject* module){

        PyImanS_HardChunkType.tp_doc = "Use HardChunk instead";
        PyImanS_HardChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_HardChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_HardChunkType.tp_init = (initproc)&PyImanS_HardChunk_Init;
        PyImanS_HardChunkType.tp_as_mapping = &PyImanS_HardChunkMapping;

        if (PyType_Ready(&PyImanS_HardChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_HardChunkType);
        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_HardChunkType;

        if (PyModule_AddObject(module, "_sourcefiles_HardChunk", (PyObject*)&PyImanS_HardChunkType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_HARDCHUNK_H
