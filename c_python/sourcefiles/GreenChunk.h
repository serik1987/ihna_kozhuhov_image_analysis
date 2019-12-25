//
// Created by serik1987 on 25.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_GREENCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_GREENCHUNK_H

extern "C" {

    typedef struct{
        PyImanS_ChunkObject super;
    } PyImanS_GreenChunkObject;

    static PyTypeObject PyImanS_GreenChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._GreenChunk",
            .tp_basicsize = sizeof(PyImanS_GreenChunkObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_GreenChunk_Init(PyImanS_GreenChunkObject* self, PyObject* args, PyObject* kwds){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "GREE", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_GreenChunk_GetProperty(PyImanS_GreenChunkObject* self, PyObject* key){
        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "The subscript index for the GREE chunk shall be a string containing "
                                             "the property name");
        }
        const char* name = PyUnicode_AsUTF8(key);
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);

        if (result != NULL){
            return result;
        } else {
            PyErr_SetString(PyExc_IndexError, "subscript index is not a valid property of the GREE chunk");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_GreenChunk_mapping = {
            .mp_length = NULL,
            .mp_subscript = (binaryfunc)PyImanS_GreenChunk_GetProperty,
            .mp_ass_subscript = NULL,
    };

    static int PyImanS_GreenChunk_Create(PyObject* module){

        PyImanS_GreenChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_GreenChunkType.tp_doc = "Use GreenChunk instead";
        PyImanS_GreenChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_GreenChunkType.tp_init = (initproc)PyImanS_GreenChunk_Init;
        PyImanS_GreenChunkType.tp_as_mapping = &PyImanS_GreenChunk_mapping;

        if (PyType_Ready(&PyImanS_GreenChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_GreenChunkType);
        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_GreenChunkType;

        if (PyModule_AddObject(module, "_sourcefiles_GreenChunk", (PyObject*)&PyImanS_GreenChunkType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_GREENCHUNK_H
