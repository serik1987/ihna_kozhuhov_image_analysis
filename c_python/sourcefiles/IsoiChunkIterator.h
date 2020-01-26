//
// Created by serik1987 on 27.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ISOICHUNKITERATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ISOICHUNKITERATOR_H

extern "C" {

    static PyImanS_IsoiChunkIteratorObject*
        PyImanS_IsoiChunkIterator_New(PyTypeObject* cls, PyObject* args, PyObject* kwds){
        auto* self = (PyImanS_IsoiChunkIteratorObject*)cls->tp_alloc(cls, 0);
        if (self != NULL){
            self->parent = NULL;
            self->handle = NULL;
        }
        return self;
    }

    static int PyImanS_IsoiChunkIterator_Init(PyImanS_IsoiChunkIteratorObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        PyImanS_IsoiChunkObject* parent;

        if (!PyArg_ParseTuple(args, "O!", &PyImanS_IsoiChunkType, &parent)){
            return -1;
        }
        self->parent = parent;
        Py_INCREF(parent);

        auto* handle = (IsoiChunk*)self->parent->super.handle;
        auto* iterator_handle = new IsoiChunk::iterator(IsoiChunk::iterator::InSoft, std::list<Chunk*>::iterator(),
                nullptr, handle);
        self->handle = iterator_handle;

        return 0;
    }

    static void PyImanS_IsoiChunkIterator_Destroy(PyImanS_IsoiChunkIteratorObject* self){
        using namespace GLOBAL_NAMESPACE;
        if (self->handle != NULL){
            delete (IsoiChunk::iterator*)self->handle;
            self->handle = NULL;
        }
        Py_XDECREF(self->parent);
        Py_TYPE(self)->tp_free(self);
    }

    static PyImanS_IsoiChunkIteratorObject* PyImanS_IsoiChunkIterator_Iter(PyImanS_IsoiChunkIteratorObject* self){
        Py_INCREF(self);
        return self;
    }

    static PyImanS_ChunkObject* PyImanS_IsoiChunkIterator_Next(PyImanS_IsoiChunkIteratorObject* self){
        using namespace GLOBAL_NAMESPACE;
        auto* it = (IsoiChunk::iterator*)self->handle;
        auto* parent = self->parent;
        auto* isoi = (IsoiChunk*)parent->super.handle;
        if (*it == isoi->end()){
            return NULL;
        }
        auto* result = &(**it);
        ++*it;
        PyImanS_ChunkObject* result_object = PyImanS_Chunk_FromHandle(result, (PyObject*)parent);

        return result_object;
    }

    static int PyImanS_IsoiChunkIterator_Create(PyObject* module){

        PyImanS_IsoiChunkIteratorType.tp_flags = Py_TPFLAGS_DEFAULT;
        PyImanS_IsoiChunkIteratorType.tp_doc = "Use this iterator in order to iterate over all chunks containing in "
                                               "the file header, inside the ISOI chunk, but not in the frame header";
        PyImanS_IsoiChunkIteratorType.tp_new = (newfunc)PyImanS_IsoiChunkIterator_New;
        PyImanS_IsoiChunkIteratorType.tp_dealloc = (destructor)PyImanS_IsoiChunkIterator_Destroy;
        PyImanS_IsoiChunkIteratorType.tp_init = (initproc)PyImanS_IsoiChunkIterator_Init;
        PyImanS_IsoiChunkIteratorType.tp_iter = (getiterfunc)&PyImanS_IsoiChunkIterator_Iter;
        PyImanS_IsoiChunkIteratorType.tp_iternext = (iternextfunc)PyImanS_IsoiChunkIterator_Next;

        if (PyType_Ready(&PyImanS_IsoiChunkIteratorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_IsoiChunkIteratorType);
        PyImanS_ChunkTypes[PyImanS_TotalChunksAdded++] = (PyObject*)&PyImanS_IsoiChunkIteratorType;

        if (PyModule_AddObject(module, "_sourcefiles_IsoiChunkIterator",
                (PyObject*)&PyImanS_IsoiChunkIteratorType) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOICHUNKITERATOR_H
