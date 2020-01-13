//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADERANDCLEANER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADERANDCLEANER_H

extern "C" {

    typedef struct {
        PyImanT_TraceReaderObject super;
    } PyImanT_TraceReaderAndCleanerObject;

    static PyTypeObject PyImanT_TraceReaderAndCleanerType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.tracereading.TraceReaderAndCleaner",
            .tp_basicsize = sizeof(PyImanT_TraceReaderAndCleanerObject),
            .tp_itemsize = 0,
    };

    static int PyImanT_TraceReaderAndCleaner_Init(PyImanT_TraceReaderAndCleanerObject* self,
            PyObject* args, PyObject* kwds){

        using namespace GLOBAL_NAMESPACE;
        PyObject* parent_train;

        if (!PyArg_ParseTuple(args, "O!", &PyImanS_StreamFileTrainType, &parent_train)){
            return -1;
        }

        Py_INCREF(parent_train);
        self->super.file_train = (PyImanS_StreamFileTrainObject*)parent_train;

        try{
            auto* train = (StreamFileTrain*)self->super.file_train->super.train_handle;
            auto* reader = new TraceReaderAndCleaner(*train);
            self->super.trace_reader_handle = reader;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static int PyImanT_TraceReaderAndCleaner_Create(PyObject* module){

        PyImanT_TraceReaderAndCleanerType.tp_base = &PyImanT_TraceReaderType;
        PyImanT_TraceReaderAndCleanerType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanT_TraceReaderAndCleanerType.tp_doc =
                "Provides a convenient interface for the trace reading and cleaning isolines from the traces \n"
                "using the Isoline object\n";
        PyImanT_TraceReaderAndCleanerType.tp_init = (initproc)PyImanT_TraceReaderAndCleaner_Init;

        if (PyType_Ready(&PyImanT_TraceReaderAndCleanerType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanT_TraceReaderAndCleanerType);
        PyImanT_TraceReaderAndCleaner_Handle = (PyObject*)&PyImanT_TraceReaderAndCleanerType;

        if (PyModule_AddObject(module, "_tracereading_TraceReaderAndCleaner",
                               (PyObject*)&PyImanT_TraceReaderAndCleanerType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADERANDCLEANER_H
