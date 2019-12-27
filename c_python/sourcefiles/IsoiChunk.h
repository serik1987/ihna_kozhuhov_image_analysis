//
// Created by serik1987 on 24.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ISOICHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ISOICHUNK_H

#include "../../cpp/source_files/IsoiChunk.h"

extern "C"{

    static PyImanS_ChunkObject* PyImanS_Chunk_FromHandle(void* handle, PyObject* parent);

    typedef struct {
        PyObject_HEAD
        PyImanS_IsoiChunkObject* parent;
        void* handle;
    } PyImanS_IsoiChunkIteratorObject;

    static PyTypeObject PyImanS_IsoiChunkIteratorType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles.IsoiChunkIterator",
            .tp_basicsize = sizeof(PyImanS_IsoiChunkIteratorObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_IsoiChunk_Init(PyImanS_IsoiChunkObject* self, PyObject* args, PyObject*){
        void* chunk_handle;
        PyObject* parent_object;

        if (PyImanS_Chunk_InitArgs(args, "ISOI", &chunk_handle, &parent_object) < 0){
            return -1;
        }

        self->super.handle = chunk_handle;
        self->super.parent = parent_object;

        return 0;
    }

    static PyObject* PyImanS_IsoiChunk_GetProperty(PyImanS_IsoiChunkObject* self, PyObject* key){
        using namespace GLOBAL_NAMESPACE;
        PyObject* result;
        const char* name;
        char chunk_name[ChunkHeader::CHUNK_ID_SIZE+1];
        IsoiChunk* isoi;
        Chunk* chunk;

        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "the subscript index for the ISOI chunk shall be string");
            return NULL;
        }
        name = PyUnicode_AsUTF8(key);
        result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);

        if (result != NULL){
            return result;
        } else {
            if (strlen(name) != ChunkHeader::CHUNK_ID_SIZE){
                PyErr_SetString(PyExc_IndexError, "The index is not a correct property of the ISOI chunk\n");
                return NULL;
            }

            try {
                for (int i = 0; i < ChunkHeader::CHUNK_ID_SIZE; ++i) {
                    chunk_name[i] = (char)toupper(name[i]);
                }
                chunk_name[ChunkHeader::CHUNK_ID_SIZE] = '\0';
                isoi = (IsoiChunk *) self->super.handle;
                chunk = isoi->getChunkById(*(uint32_t *) chunk_name);

                if (chunk == nullptr){
                    char S[128];
                    sprintf(S, "Chunk '%s' doesn't exist or is not at the file header", chunk_name);
                    PyErr_SetString(PyExc_IndexError, S);
                    return NULL;
                }

                return (PyObject*)PyImanS_Chunk_FromHandle(chunk, (PyObject*)self);
            } catch (std::exception& e) {
                PyIman_Exception_process(&e);
                return NULL;
            }
        }
    }

    static PyMappingMethods PyImanS_IsoiChunk_mapping = {
            .mp_length = NULL,
            .mp_subscript = (binaryfunc)PyImanS_IsoiChunk_GetProperty,
            .mp_ass_subscript = NULL,
    };

    static PyImanS_IsoiChunkIteratorObject* PyImanS_IsoiChunk_Iter(PyImanS_IsoiChunkObject* self){
        return (PyImanS_IsoiChunkIteratorObject*)
            PyObject_CallFunction((PyObject*)&PyImanS_IsoiChunkIteratorType, "O", self);
    }

    static int PyImanS_IsoiChunk_Create(PyObject* module){

        PyImanS_IsoiChunkType.tp_doc = "Use IsoiChunk instead";
        PyImanS_IsoiChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_IsoiChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_IsoiChunkType.tp_init = (initproc)PyImanS_IsoiChunk_Init;
        PyImanS_IsoiChunkType.tp_as_mapping = &PyImanS_IsoiChunk_mapping;
        PyImanS_IsoiChunkType.tp_iter = (getiterfunc)PyImanS_IsoiChunk_Iter;

        if (PyType_Ready(&PyImanS_IsoiChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_IsoiChunkType);

        if (PyModule_AddObject(module, "_sourcefiles_IsoiChunk", (PyObject*)&PyImanS_IsoiChunkType) < 0){
            Py_DECREF(&PyImanS_IsoiChunkType);
            return -1;
        }

        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_IsoiChunkType;

        return 0;
    }

};


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOICHUNK_H
