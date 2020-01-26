//
// Created by serik1987 on 24.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPCHUNK_H

extern "C"{

    typedef struct {
        PyImanS_ChunkObject super;
    } PyImanS_CompChunkObject;

    static PyTypeObject PyImanS_CompChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.sourcefiles._CompChunk",
            /* tp_basicsize */ sizeof(PyImanS_CompChunkObject),
            /* tp_itemsize */ 0,
    };

    static int PyImanS_CompChunk_Init(PyImanS_CompChunkObject* self, PyObject* args, PyObject*){
        void* handle;
        PyObject* parent;

        if (PyImanS_Chunk_InitArgs(args, "COMP", &handle, &parent) < 0){
            return -1;
        }

        self->super.handle = handle;
        self->super.parent = parent;

        return 0;
    }

    static PyObject* PyImanS_CompChunk_GetProperty(PyImanS_CompChunkObject* self, PyObject* key){
        using namespace GLOBAL_NAMESPACE;

        if (!PyUnicode_Check(key)){
            PyErr_SetString(PyExc_TypeError, "subscript indices for this object must be strings only");
            return NULL;
        }
        const char* name = PyUnicode_AsUTF8(key);
        PyObject* result = PyImanS_Chunk_GetProperty((PyImanS_ChunkObject*)self, name);
        auto* chunk = (CompChunk*)self->super.handle;

        if (result != NULL) {
            return result;
        } else if (strcmp(name, "compressed_record_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCompressedRecordSize());
        } else if (strcmp(name, "compressed_frame_size") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCompressedFrameSize());
        } else if (strcmp(name, "compressed_frame_number") == 0) {
            return PyLong_FromUnsignedLong(chunk->getCompressedFrameNumber());
        } else {
            PyErr_SetString(PyExc_IndexError, "subscript index doesn't refer to the valid COMP chunk property");
            return NULL;
        }
    }

    static PyMappingMethods PyImanS_CompChunk_mapping = {
            /* mp_length */ NULL,
            /* mp_subscript */ (binaryfunc)PyImanS_CompChunk_GetProperty,
            /* mp_ass_subscript */ NULL,
    };

    static int PyImanS_CompChunk_Create(PyObject* module){

        PyImanS_CompChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_CompChunkType.tp_doc = "Use CompChunk instead";
        PyImanS_CompChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_CompChunkType.tp_init = (initproc)PyImanS_CompChunk_Init;
        PyImanS_CompChunkType.tp_as_mapping = &PyImanS_CompChunk_mapping;

        if (PyType_Ready(&PyImanS_CompChunkType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_CompChunkType);

        if (PyModule_AddObject(module, "_sourcefiles_CompChunk", (PyObject*)&PyImanS_CompChunkType) < 0){
            Py_DECREF(&PyImanS_CompChunkType);
            return -1;
        }

        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_CompChunkType;

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPCHUNK_H
