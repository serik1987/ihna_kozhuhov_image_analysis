//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADERANDCLEANER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADERANDCLEANER_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        void* parent_train;
        void* parent_synchronization;
        void* isoline_handle;
    } PyImanI_IsolineObject;

    static PyTypeObject PyImanI_IsolineType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.isolines.Isoline",
            .tp_basicsize = sizeof(PyImanI_IsolineObject),
            .tp_itemsize = 0,
    };

    typedef struct {
        PyImanT_TraceReaderObject super;
        PyObject* cleaner;
    } PyImanT_TraceReaderAndCleanerObject;

    static PyTypeObject PyImanT_TraceReaderAndCleanerType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.tracereading.TraceReaderAndCleaner",
            .tp_basicsize = sizeof(PyImanT_TraceReaderAndCleanerObject),
            .tp_itemsize = 0,
    };

    static PyImanT_TraceReaderAndCleanerObject* PyImanT_TraceReaderAndCleaner_New
        (PyTypeObject* type, PyObject* arg, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* self = (PyImanT_TraceReaderAndCleanerObject*)PyImanT_TraceReader_New(type, arg, kwds);
        if (self != NULL){
            self->cleaner = NULL;
        }
        return self;
    }

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

    static void PyImanT_TraceReaderAndCleaner_Destroy(PyImanT_TraceReaderAndCleanerObject* self){
        Py_XDECREF(self->cleaner);
        PyImanT_TraceReader_Destroy((PyImanT_TraceReaderObject*)self);
    }

    static PyObject* PyImanI_TraceReaderAndCleaner_HasRead(PyImanT_TraceReaderAndCleanerObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReaderAndCleaner*)self->super.trace_reader_handle;

        try{
            return PyBool_FromLong(reader->isCleaned());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReaderAndCleaner_TracesBeforeRemove
            (PyImanT_TraceReaderAndCleanerObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReaderAndCleaner*)self->super.trace_reader_handle;


        try{
            const double* traces = reader->getTracesBeforeRemove();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReaderAndCleaner_Isolines
            (PyImanT_TraceReaderAndCleanerObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReaderAndCleaner*)self->super.trace_reader_handle;

        try{
            const double* isolines = reader->getIsolines();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanI_TraceReaderAndCleaner_SetIsolineRemover(
            PyImanT_TraceReaderAndCleanerObject* self, PyObject* isoline_object, void*){
        using namespace GLOBAL_NAMESPACE;

        if (PyObject_IsInstance(isoline_object, (PyObject*)&PyImanI_IsolineType) != 1){
            PyErr_SetString(PyExc_ValueError, "Bad value for isoline_remover property");
            return -1;
        }

        self->cleaner = isoline_object;
        Py_INCREF(self->cleaner);

        auto* reader = (TraceReaderAndCleaner*)self->super.trace_reader_handle;

        try{
            auto* remover_object = (PyImanI_IsolineObject*)self->cleaner;
            auto* remover = (Isoline*)remover_object->isoline_handle;
            reader->setIsolineRemover(*remover);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }

        return 0;
    }

    static PyGetSetDef PyImanI_TraceReaderAndCleaner_Properties[] = {
            {(char*)"has_cleaned", (getter)PyImanI_TraceReaderAndCleaner_HasRead, NULL,
                    (char*)"True if the traces have been read and cleaned, False otherwise"},

            {(char*)"traces_before_remove", (getter)PyImanT_TraceReaderAndCleaner_TracesBeforeRemove, NULL,
             (char*)"Array containing traces before isolines were removed Designations are the same as in traces"},

            {(char*)"isolines", (getter)PyImanT_TraceReaderAndCleaner_Isolines, NULL,
             (char*)"Array containing isolines. Designations are the same as in traces"},

            {(char*)"isoline_remover", NULL, (setter)PyImanI_TraceReaderAndCleaner_SetIsolineRemover,
             (char*)"An instance of Isoline object that is responsible for isoline removing"},

            {NULL}
    };

    static int PyImanT_TraceReaderAndCleaner_Create(PyObject* module){

        PyImanT_TraceReaderAndCleanerType.tp_base = &PyImanT_TraceReaderType;
        PyImanT_TraceReaderAndCleanerType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanT_TraceReaderAndCleanerType.tp_doc =
                "Provides a convenient interface for the trace reading and cleaning isolines from the traces \n"
                "using the Isoline object\n";
        PyImanT_TraceReaderAndCleanerType.tp_new = (newfunc)PyImanT_TraceReaderAndCleaner_New;
        PyImanT_TraceReaderAndCleanerType.tp_init = (initproc)PyImanT_TraceReaderAndCleaner_Init;
        PyImanT_TraceReaderAndCleanerType.tp_dealloc = (destructor)PyImanT_TraceReaderAndCleaner_Destroy;
        PyImanT_TraceReaderAndCleanerType.tp_getset = PyImanI_TraceReaderAndCleaner_Properties;

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
