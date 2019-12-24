//
// Created by serik1987 on 24.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SOFTCHUNK_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SOFTCHUNK_H

extern "C" {

    typedef struct{
        PyImanS_ChunkObject super;
    } PyImanS_SoftChunkObject;

    static PyTypeObject PyImanS_SoftChunkType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._SoftChunk",
            .tp_basicsize = sizeof(PyImanS_SoftChunkObject),
            .tp_itemsize = 0,
    };

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

    static int PyImanS_SoftChunk_Create(PyObject* module){

        PyImanS_SoftChunkType.tp_base = &PyImanS_ChunkType;
        PyImanS_SoftChunkType.tp_doc = "Use SoftChunk instead";
        PyImanS_SoftChunkType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_SoftChunkType.tp_init = (initproc)PyImanS_SoftChunk_Init;

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

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SOFTCHUNK_H
