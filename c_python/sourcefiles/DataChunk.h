//
// Created by serik1987 on 25.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_DATACHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_DATACHUNK_H

extern "C" {

    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_DataChunkObject;

    static PyTypeObject PyImanS_DataChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._DataChunk",
            .tp_basicsize = sizeof(PyImanS_DataChunkObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_DataChunk_Init(PyImanS_DataChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "DATA", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_DataChunk_GetProperty(PyImanS_DataChunkObject* self, PyObject* key){
        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "subscript index for the DATA chunk shall be a string containing a "
                                             "property name");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);

        if (result != NULL){
            return result;
        } else {
            PyErr_SetString(PyExc_IndexError, "The key doesn't represent a valid property of the data chunk");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_DataChunk_mapping = {
            .mp_length = NULL,
            .mp_subscript = (binaryfunc)&PyImanS_DataChunk_GetProperty,
            .mp_ass_subscript = NULL,
    };

    static int PyImanS_DataChunk_Create(PyObject* module){

        PyImanS_DataChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_DataChunkType.tp_doc = "Use DataChunk instead";
        PyImanS_DataChunkType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanS_DataChunkType.tp_init = (initproc)PyImanS_DataChunk_Init;
        PyImanS_DataChunkType.tp_as_mapping = &PyImanS_DataChunk_mapping;

        if (PyType_Ready(&PyImanS_DataChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_DataChunkType);
        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_DataChunkType;

        if (PyModule_AddObject(module, "_sourcefiles_DataChunk", (PyObject*)&PyImanS_DataChunkType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DATACHUNK_H
