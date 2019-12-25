//
// Created by serik1987 on 25.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COSTCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COSTCHUNK_H

#include "../../cpp/source_files/CostChunk.h"

extern "C" {
    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_CostChunkObject;

    static PyTypeObject PyImanS_CostChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._CostChunk",
            .tp_basicsize = sizeof(PyImanS_CostChunkObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_CostChunk_Init(PyImanS_CostChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "COST", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_CostChunk_GetProperty(PyImanS_CostChunkObject* self, PyObject* key){
        using namespace GLOBAL_NAMESPACE;

        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "Subscript indices must be strings containing property names");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);
        auto* chunk = (CostChunk*)self->super.handle;

        if (result != NULL) {
            return result;
        } else if (strcmp(name, "synchronization_channel_number") == 0) {
            return PyLong_FromLong(chunk->getSynchronizationChannels());
        } else if (strcmp(name, "synchronization_channel_max") == 0) {
            result = PyTuple_New(chunk->getSynchronizationChannels());
            if (result == NULL) return NULL;
            for (int chan = 0; chan < chunk->getSynchronizationChannels(); ++chan) {
                uint32_t max = chunk->getSynchronizationChannelsMax(chan);
                PyObject *maxValue = PyLong_FromUnsignedLong(max);
                if (PyTuple_SetItem(result, chan, maxValue) < 0) {
                    Py_DECREF(result);
                    return NULL;
                }
            }
            return result;
        } else if (strcmp(name, "stimulus_channels") == 0) {
            return PyLong_FromLong(chunk->getStimulusChannel());
        } else if (strcmp(name, "stimulus_period") == 0){
            result = PyTuple_New(chunk->getStimulusChannel());
            if (result == NULL) return NULL;
            for (int chan = 0; chan < chunk->getStimulusChannel(); ++chunk){
                uint32_t period = chunk->getStimulusPeriod(chan);
                PyObject* periodValue = PyLong_FromUnsignedLong(period);
                if (PyTuple_SetItem(result, chan, periodValue) < 0){
                    Py_DECREF(result);
                    return NULL;
                }
            }
            return result;
        } else {
            PyErr_SetString(PyExc_IndexError, "The key doesn't refer to the correct property of the COST chunk");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_CostChunk_subscripts = {
        .mp_length = NULL,
        .mp_subscript = (binaryfunc)PyImanS_CostChunk_GetProperty,
        .mp_ass_subscript = NULL,
    };

    static int PyImanS_CostChunk_Create(PyObject* module){

        PyImanS_CostChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_CostChunkType.tp_doc = "Use CostChunk instead";
        PyImanS_CostChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_CostChunkType.tp_init = (initproc)PyImanS_CostChunk_Init;
        PyImanS_CostChunkType.tp_as_mapping = &PyImanS_CostChunk_subscripts;

        if (PyType_Ready(&PyImanS_CostChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_CostChunkType);

        if (PyModule_AddObject(module, "_sourcefiles_CostChunk", (PyObject*)&PyImanS_CostChunkType) < 0){
            Py_DECREF(&PyImanS_CostChunkType);
            return -1;
        }

        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_CostChunkType;

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COSTCHUNK_H
