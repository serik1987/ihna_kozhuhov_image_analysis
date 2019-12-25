//
// Created by serik1987 on 25.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_EPSTCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_EPSTCHUNK_H

#include "../../cpp/source_files/EpstChunk.h"

extern "C" {

    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_EpstChunkObject;

    static PyTypeObject PyImanS_EpstChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._EpstChunk",
            .tp_basicsize = sizeof(PyImanS_EpstChunkObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_EpstChunk_Init(PyImanS_EpstChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "EPST", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_EpstChunk_GetProperty(PyImanS_EpstChunkObject* self, PyObject* key){
        using namespace GLOBAL_NAMESPACE;

        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "The subscript index for the EPST chunk shall be a string containing "
                                             "property name");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);
        auto* chunk = (EpstChunk*)self->super.handle;

        if (result != NULL) {
            return result;
        } else if (strcmp(name, "condition_number") == 0) {
            return PyLong_FromUnsignedLong(chunk->getConditionNumber());
        } else if (strcmp(name, "repetition_number") == 0) {
            return PyLong_FromUnsignedLong(chunk->getRepetitionsNumber());
        } else if (strcmp(name, "randomized") == 0) {
            return PyBool_FromLong(chunk->getRandomize());
        } else if (strcmp(name, "iti_frames") == 0) {
            return PyLong_FromUnsignedLong(chunk->getItiFrames());
        } else if (strcmp(name, "stimulus_frames") == 0) {
            return PyLong_FromUnsignedLong(chunk->getStimulusFrames());
        } else if (strcmp(name, "pre_frames") == 0) {
            return PyLong_FromUnsignedLong(chunk->getPrestimulusFrames());
        } else if (strcmp(name, "post_frames") == 0) {
            return PyLong_FromUnsignedLong(chunk->getPoststimulusFrames());
        } else {
            PyErr_SetString(PyExc_IndexError, "No such property in the EPST chunk");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_EpstChunk_mapping = {
            .mp_length = NULL,
            .mp_subscript = (binaryfunc)PyImanS_EpstChunk_GetProperty,
            .mp_ass_subscript = NULL,
    };

    static int PyImanS_EpstChunk_Create(PyObject* module){

        PyImanS_EpstChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_EpstChunkType.tp_doc = "Use EpstChunk instead";
        PyImanS_EpstChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_EpstChunkType.tp_init = (initproc)PyImanS_EpstChunk_Init;
        PyImanS_EpstChunkType.tp_as_mapping = &PyImanS_EpstChunk_mapping;

        if (PyType_Ready(&PyImanS_EpstChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_EpstChunkType);
        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_EpstChunkType;

        if (PyModule_AddObject(module, "_sourcefiles_EpstChunk", (PyObject*)&PyImanS_EpstChunkType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EPSTCHUNK_H
