//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSEDFILETRAINITERATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSEDFILETRAINITERATOR_H

extern "C" {

    static PyImanS_CompressedFileTrainIteratorObject*
        PyImanS_CompressedFileTrainIterator_New(PyTypeObject* cls, PyObject* args, PyObject* kwds){
        auto* self = (PyImanS_CompressedFileTrainIteratorObject*)
                PyImanS_FileTrainIterator_New(cls, args, kwds);
        return self;
    }

    static int PyImanS_CompressedFileTrainIterator_Init
            (PyImanS_CompressedFileTrainIteratorObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        PyObject* parent = NULL;

        if (!PyArg_ParseTuple(args, "O!", &PyImanS_CompressedFileTrainType, &parent)){
            return -1;
        }

        Py_INCREF(parent);
        self->super.parent_train = parent;

        auto* train_object = (PyImanS_CompressedFileTrainObject*)parent;
        auto* train = (CompressedFileTrain*)train_object->super.train_handle;
        auto* it = new std::list<TrainSourceFile*>::iterator();
        *it = train->begin();
        self->super.iterator_handle = it;

        return 0;
    }

    static PyImanS_CompressedSourceFileObject*
        PyImanS_CompressedSourceFileIterator_Next(PyImanS_CompressedFileTrainIteratorObject* self){
        using namespace GLOBAL_NAMESPACE;
        auto* cpp_iterator = (std::list<TrainSourceFile*>::iterator*)self->super.iterator_handle;
        auto* parent = (PyImanS_CompressedFileTrainObject*)self->super.parent_train;
        auto* train = (CompressedFileTrain*)parent->super.train_handle;

        if (*cpp_iterator == train->end()){
            return NULL;
        }

        auto* result = (PyImanS_CompressedSourceFileObject*) PyObject_CallFunction(
                (PyObject*)&PyImanS_CompressedSourceFileType, "sssO", "some-path", "some-file", "ignore", parent);
        result->super.super.file_handle = **cpp_iterator;

        ++*cpp_iterator;

        return result;
    }

    static int PyImanS_CompressedFileTrainIterator_Create(PyObject* module){

        PyImanS_CompressedFileTrainIteratorType.tp_base = &PyImanS_FileTrainIteratorType;
        PyImanS_CompressedFileTrainIteratorType.tp_doc =
                "Allows to iterate over the compressed file train and consequtively access to all files within "
                "the train";
        PyImanS_CompressedFileTrainIteratorType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_CompressedFileTrainIteratorType.tp_new = (newfunc)PyImanS_CompressedFileTrainIterator_New;
        PyImanS_CompressedFileTrainIteratorType.tp_init = (initproc)PyImanS_CompressedFileTrainIterator_Init;
        PyImanS_CompressedFileTrainIteratorType.tp_iternext = (iternextfunc)PyImanS_CompressedSourceFileIterator_Next;

        if (PyType_Ready(&PyImanS_CompressedFileTrainIteratorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_CompressedFileTrainIteratorType);

        if (PyModule_AddObject(module, "_sourcefiles_CompressedFileTrainIterator",
                               (PyObject*)&PyImanS_CompressedFileTrainIteratorType) < 0){
            Py_DECREF(&PyImanS_CompressedFileTrainIteratorType);
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAINITERATOR_H
