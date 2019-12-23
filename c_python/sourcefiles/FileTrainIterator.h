//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FILETRAINITERATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FILETRAINITERATOR_H

extern "C" {

    static PyImanS_FileTrainIteratorObject* PyImanS_FileTrainIterator_New(PyTypeObject* cls, PyObject*, PyObject*){
        printf("New file train iterator...\n");
        PyImanS_FileTrainIteratorObject* self;
        self = (PyImanS_FileTrainIteratorObject*)cls->tp_alloc(cls, 0);
        if (self != NULL){
            self->iterator_handle = NULL;
            self->parent_train = NULL;
        }
        return self;
    }

    static int PyImanS_FileTrainIterator_Init(PyImanS_FileTrainIteratorObject*, PyObject*, PyObject*){
        PyErr_SetString(PyExc_NotImplementedError,
                "ihna.kozhukhov.imageanalysis.sourcefiles.FileTrainIterator class is purely abstract");
        return -1;
    }

    static void PyImanS_FileTrainIterator_Destroy(PyImanS_FileTrainIteratorObject* self){
        using namespace GLOBAL_NAMESPACE;

        if (self->iterator_handle != NULL){
            printf("Destruction of the C++ file train iterator handle...\n");
            auto* iterator = (std::list<TrainSourceFile*>::iterator*)self->iterator_handle;
            delete iterator;
            self->iterator_handle = NULL;
        }
        Py_XDECREF(self->parent_train);
        Py_TYPE(self)->tp_free(self);
    }

    static PyObject* PyImanS_FileTrainIterator_Iter(PyObject* self){
        Py_INCREF(self);
        return self;
    }

    static PyTypeObject PyImanS_FileTrainIteratorType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles.FileTrainIterator",
            .tp_basicsize = sizeof(PyImanS_FileTrainIteratorObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_FileTrainIterator_Create(PyObject* module){

        PyImanS_FileTrainIteratorType.tp_doc = "This is the base class for all file train iterators";
        PyImanS_FileTrainIteratorType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_FileTrainIteratorType.tp_new = (newfunc)PyImanS_FileTrainIterator_New;
        PyImanS_FileTrainIteratorType.tp_dealloc = (destructor)PyImanS_FileTrainIterator_Destroy;
        PyImanS_FileTrainIteratorType.tp_init = (initproc)PyImanS_FileTrainIterator_Init;
        PyImanS_FileTrainIteratorType.tp_iter = (getiterfunc)PyImanS_FileTrainIterator_Iter;

        if (PyType_Ready(&PyImanS_FileTrainIteratorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_FileTrainIteratorType);

        if (PyModule_AddObject(module, "_sourcefiles_FileTrainIterator",
                (PyObject*)&PyImanS_FileTrainIteratorType) < 0){
            Py_DECREF(&PyImanS_FileTrainIteratorType);
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAINITERATOR_H
