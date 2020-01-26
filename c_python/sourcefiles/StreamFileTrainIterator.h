//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAINITERATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAINITERATOR_H

extern "C" {

    static PyImanS_StreamFileTrainIteratorObject* PyImanS_StreamFileTrainIterator_New
            (PyTypeObject* cls, PyObject* args, PyObject* kwds){
        auto* self = (PyImanS_StreamFileTrainIteratorObject*)PyImanS_FileTrainIterator_New(cls, args, kwds);
        return self;
    }

    static int PyImanS_StreamFileTrainIterator_Init(PyImanS_StreamFileTrainIteratorObject* self,
            PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        PyObject* parent;

        if (!PyArg_ParseTuple(args, "O", &parent)){
            return -1;
        }

        int is_instance = PyObject_IsInstance(parent, (PyObject*)&PyImanS_StreamFileTrainType);
        if (is_instance == 0){
            PyErr_SetString(PyExc_ValueError, "The argument shall be an instance of FileTrain");
        }
        if (is_instance != 1){
            return -1;
        }

        self->super.parent_train = parent;
        Py_INCREF(parent);

        auto* train_object = (PyImanS_StreamFileTrainObject*)parent;
        auto* train = (StreamFileTrain*)train_object->super.train_handle;
        auto* iterator_handle = new std::list<TrainSourceFile*>::iterator();
        *iterator_handle = train->begin();
        self->super.iterator_handle = iterator_handle;

        return 0;
    }

    static PyImanS_StreamSourceFileObject* PyImanS_StreamFileTrainIterator_Next
        (PyImanS_StreamFileTrainIteratorObject* self){
        using namespace GLOBAL_NAMESPACE;
        auto* cpp_iterator = (std::list<TrainSourceFile*>::iterator*)self->super.iterator_handle;
        auto* parent = (PyImanS_FileTrainObject*)self->super.parent_train;
        auto* train  = (FileTrain*)parent->train_handle;
        if (*cpp_iterator == train->end()){
            return NULL;
        }
        auto* source_file = **cpp_iterator;
        ++*cpp_iterator;

        auto* file = (PyImanS_StreamSourceFileObject*)PyObject_CallFunction(
                (PyObject*)&PyImanS_StreamSourceFileType,
                "sssO", "some-path", "some-file", "ignore", parent);
        file->super.super.file_handle = source_file;

        return file;
    }

    static int PyImanS_StreamFileTrainIterator_Create(PyObject* module){

        PyImanS_SourceFileTrainIteratorType.tp_doc =
                "The iterator that allows to iterate over all files containing in the StreamFileTrain\n"
                "\n"
                "Create arguments:\n"
                "\tparent - the file stream file train to be initialized\n";
        PyImanS_SourceFileTrainIteratorType.tp_base = &PyImanS_FileTrainIteratorType;
        PyImanS_SourceFileTrainIteratorType.tp_new = (newfunc)PyImanS_StreamFileTrainIterator_New;
        PyImanS_SourceFileTrainIteratorType.tp_init = (initproc)PyImanS_StreamFileTrainIterator_Init;
        PyImanS_SourceFileTrainIteratorType.tp_iternext = (iternextfunc)PyImanS_StreamFileTrainIterator_Next;

        if (PyType_Ready(&PyImanS_SourceFileTrainIteratorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_SourceFileTrainIteratorType);

        if (PyModule_AddObject(module, "_sourcefiles_StreamFileTrainIterator",
                               (PyObject*)&PyImanS_SourceFileTrainIteratorType) < 0){
            Py_DECREF(&PyImanS_SourceFileTrainIteratorType);
            return -1;
        }


        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAINITERATOR_H
