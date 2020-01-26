//
// Created by serik1987 on 26.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SYNCCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SYNCCHUNK_H

#include "../../cpp/source_files/SyncChunk.h"

extern "C" {

    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_SyncChunkObject;

    static PyTypeObject PyImanS_SyncChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            "ihna.kozhukhov.imageanalysis.sourcefiles._SyncChunk",
            sizeof(PyImanS_SyncChunkObject),
            0,
    };

    static int PyImanS_SyncChunk_Init(PyImanS_SyncChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "SYNC", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_SyncChunk_GetProperty(PyImanS_SyncChunkObject* self, PyObject* key){
        using namespace GLOBAL_NAMESPACE;

        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "subscript index for the SYNC chunk shall be a string containing "
                                             "a property name");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        // auto* chunk = (SyncChunk*)self->super.handle;
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);

        if (result != NULL){
            return result;
        } else {
            PyErr_SetString(PyExc_IndexError, "subscript index for the SYNC chunk doesn't refer to the valid property");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_SynChunkMethods = {
            /* mp_length */ NULL,
            /* mp_subscript */ (binaryfunc)PyImanS_SyncChunk_GetProperty,
            /* mp_ass_subscript */ NULL,
    };

    static int PyImanS_SyncChunk_Create(PyObject *module) {

        PyImanS_SyncChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_SyncChunkType.tp_doc = "Use SyncChunk instead";
        PyImanS_SyncChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_SyncChunkType.tp_init = (initproc)&PyImanS_SyncChunk_Init;
        PyImanS_SyncChunkType.tp_as_mapping = &PyImanS_SynChunkMethods;

        if (PyType_Ready(&PyImanS_SyncChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_SyncChunkType);
        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_SyncChunkType;

        if (PyModule_AddObject(module, "_sourcefiles_SyncChunk", (PyObject*)&PyImanS_SyncChunkType) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SYNCCHUNK_H
