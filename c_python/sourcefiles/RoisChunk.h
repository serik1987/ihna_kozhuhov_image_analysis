//
// Created by serik1987 on 26.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ROISCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ROISCHUNK_H

#include "../../cpp/source_files/RoisChunk.h"

extern "C" {

    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_RoisChunkObject;

    static PyTypeObject PyImanS_RoisChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._RoisChunk",
            .tp_basicsize = sizeof(PyImanS_RoisChunkObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_RoisChunk_Init(PyImanS_RoisChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "ROIS", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_RoisChunk_GetProperty(PyImanS_RoisChunkObject* self, PyObject* key) {
        using namespace GLOBAL_NAMESPACE;
        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "Chunk subscript indices shall be strings containing property names");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        auto* pchunk = (RoisChunk*)self->super.handle;
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);

        if (result != NULL){
            return result;
        } else {
            PyErr_SetString(PyExc_IndexError, "Subscript index for the ROIS chunk referes to unknown or unsupported "
                                              "property");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_RoisChunkMapping = {
            .mp_length = NULL,
            .mp_subscript = (binaryfunc)PyImanS_RoisChunk_GetProperty,
            .mp_ass_subscript = NULL,
    };

    static int PyImanS_RoiChunk_Create(PyObject* module){

        PyImanS_RoisChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_RoisChunkType.tp_doc = "Use RoisChunk instead";
        PyImanS_RoisChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_RoisChunkType.tp_init = (initproc)&PyImanS_RoisChunk_Init;
        PyImanS_RoisChunkType.tp_as_mapping = &PyImanS_RoisChunkMapping;

        if (PyType_Ready(&PyImanS_RoisChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_RoisChunkType);
        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_RoisChunkType;

        if (PyModule_AddObject(module, "_sourcefiles_RoisChunk", (PyObject*)&PyImanS_RoisChunkType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ROISCHUNK_H
