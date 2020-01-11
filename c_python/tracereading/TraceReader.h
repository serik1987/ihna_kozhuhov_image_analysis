//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADER_H

#include "../../cpp/tracereading/TraceReader.h"

extern "C" {

    typedef struct {
        PyObject_HEAD
        void* trace_reader_handle;
        PyImanS_StreamFileTrainObject* file_train;
    } PyImanT_TraceReaderObject;

    static PyTypeObject PyImanT_TraceReaderType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.tracereader.TraceReader",
            .tp_basicsize = sizeof(PyImanT_TraceReaderObject),
            .tp_itemsize = 0,
    };

    static PyImanT_TraceReaderObject* PyImanT_TraceReader_New(PyTypeObject* type, PyObject* arg, PyObject* kwds){
        PyImanT_TraceReaderObject* self = NULL;
        self = (PyImanT_TraceReaderObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->trace_reader_handle = NULL;
            self->file_train = NULL;
        }
        return self;
    }

    static int PyImanT_TraceReader_Init(PyImanT_TraceReaderObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        if (!PyArg_ParseTuple(args, "O!", &PyImanS_StreamFileTrainType, &self->file_train)){
            return -1;
        }

        Py_INCREF(self->file_train);

        auto* train = (StreamFileTrain*)self->file_train->super.train_handle;
        self->trace_reader_handle = new TraceReader(*train);

        return 0;
    }

    static void PyImanT_TraceReader_Destroy(PyImanT_TraceReaderObject* self){
        using namespace GLOBAL_NAMESPACE;

        if (self->trace_reader_handle != NULL){
            auto* reader = (TraceReader*)self->trace_reader_handle;
            delete reader;
        }

        Py_XDECREF(self->file_train);

        Py_TYPE(self)->tp_free(self);
    }

    int PyImanT_TraceReader_Create(PyObject* module){

        PyImanT_TraceReaderType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanT_TraceReaderType.tp_doc = "This is an engine to read the temporal-dependent signal from a certain \n"
                                         "map pixel, sychronization channel or arrival timestamp\n"
                                         "\n"
                                         "Usage: trace = TraceReader(train)\n"
                                         "where train is an instance of StreamFileTrain";
        PyImanT_TraceReaderType.tp_new = (newfunc)PyImanT_TraceReader_New;
        PyImanT_TraceReaderType.tp_init = (initproc)PyImanT_TraceReader_Init;
        PyImanT_TraceReaderType.tp_dealloc = (destructor)PyImanT_TraceReader_Destroy;

        if (PyType_Ready(&PyImanT_TraceReaderType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanT_TraceReaderType);

        if (PyModule_AddObject(module, "_tracereading_TraceReader", (PyObject*)&PyImanT_TraceReaderType) < 0){
            Py_DECREF(&PyImanT_TraceReaderType);
            return -1;
        }

        PyImanT_TraceReaderTypeHandle = (PyObject*)&PyImanT_TraceReaderType;

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H
