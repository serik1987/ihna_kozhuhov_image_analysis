//
// Created by serik1987 on 08.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSOR_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        PyObject* source_train;
        void* compressor_handle;
        PyObject* progress_bar;
    } PyImanC_CompressorObject;

    static PyTypeObject PyImanC_CompressorType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.compression._Compressor",
            /* tp_basicsize */ sizeof(PyImanC_CompressorObject),
            /* tp_itemsize */ 0,
    };

    static PyImanC_CompressorObject* PyImanC_Compressor_New(PyTypeObject* type, PyObject*, PyObject){
        printf("SO Creating new compressor object\n");
        PyImanC_CompressorObject* self;
        self = (PyImanC_CompressorObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->source_train = NULL;
            self->compressor_handle = NULL;
            self->progress_bar = NULL;
        }
        return self;
    }

    static int PyImanC_Compressor_Init(PyImanC_CompressorObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        const char* output_dir;
        printf("SO Initialization of the compressor\n");

        if (!PyArg_ParseTuple(args, "O!s", (PyObject*)&PyImanS_StreamFileTrainType, &self->source_train, &output_dir)){
            return -1;
        }

        Py_INCREF(self->source_train);

        auto* train_object = (PyImanS_StreamFileTrainObject*)self->source_train;
        auto* train = (StreamFileTrain*)train_object->super.train_handle;

        try{
                auto* compressor = new Compressor(*train, output_dir);
                self->compressor_handle = compressor;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }

        return 0;
    }

    static PyObject* PyImanC_Compressor_Run(PyImanC_CompressorObject* self, PyObject*, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        auto* compressor = (Compressor*)self->compressor_handle;

        try{
            compressor->run();
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

        return Py_BuildValue("");
    }

    static void PyImanC_Compressor_Progress(float perc, void* handle){
        auto* compressor_object = (PyImanC_CompressorObject*)handle;
        auto* f = compressor_object->progress_bar;
        auto* result = PyObject_CallFunction(f, "f", perc);
        if (result != NULL){
            Py_DECREF(result);
        } else {
            printf("Compressor::progress_bar: fail to run the progress bar function\n");
        }
    }

    static PyObject* PyImanC_Compressor_SetProgress(PyImanC_CompressorObject* self, PyObject* f, PyObject*){
        using namespace GLOBAL_NAMESPACE;

        PyObject* result = PyObject_CallFunction(f, "f", 0.0);
        if (result == NULL){
            return NULL;
        }
        Py_DECREF(result);

        Py_XDECREF(self->progress_bar);
        self->progress_bar = f;
        Py_INCREF(f);

        auto* compressor = (Compressor*)self->compressor_handle;
        try{
            compressor->setProgressFunction(PyImanC_Compressor_Progress, self);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

        printf("Number of references: %ld\n", Py_REFCNT(f));

        return Py_BuildValue("");
    }

    static PyObject* PyImanC_Compressor_GetOutputTrainName(PyImanC_CompressorObject* self, PyObject*, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        auto* compressor = (Compressor*)self->compressor_handle;
        return Py_BuildValue("s", compressor->getFullOutputFile().c_str());
    }

    static PyMethodDef PyImanC_Compressor_methods[] = {
            {"run", (PyCFunction)PyImanC_Compressor_Run, METH_NOARGS, ""},
            {"set_progress_bar", (PyCFunction)PyImanC_Compressor_SetProgress, METH_O, ""},
            {"get_output_train_name", (PyCFunction)PyImanC_Compressor_GetOutputTrainName, METH_NOARGS, ""},
            {NULL}
    };

    static void PyImanC_Compressor_Destroy(PyImanC_CompressorObject* self){
        printf("SO Compressor destruction\n");
        using namespace GLOBAL_NAMESPACE;

        Py_XDECREF(self->source_train);
        printf("SO Source train destroyed\n");
        Py_XDECREF(self->progress_bar);
        printf("SO Progress bar destroyed\n");

        if (self->compressor_handle != NULL){
            auto* compressor = (Compressor*)self->compressor_handle;
            delete compressor;
        }

        Py_TYPE(self)->tp_free(self);
    }

    static int PyImanC_Compressor_Create(PyObject *module) {

        PyImanC_CompressorType.tp_doc = "Use compress(...) instead";
        PyImanC_CompressorType.tp_flags = Py_TPFLAGS_DEFAULT;
        PyImanC_CompressorType.tp_new = (newfunc)PyImanC_Compressor_New;
        PyImanC_CompressorType.tp_dealloc = (destructor)PyImanC_Compressor_Destroy;
        PyImanC_CompressorType.tp_init = (initproc)PyImanC_Compressor_Init;
        PyImanC_CompressorType.tp_methods = PyImanC_Compressor_methods;

        if (PyType_Ready(&PyImanC_CompressorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanC_CompressorType);
        PyImanC_Class_list[PyImanC_Current_class++] = (PyObject*)&PyImanC_CompressorType;

        if (PyModule_AddObject(module, "_compression_Compressor", (PyObject*)&PyImanC_CompressorType) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSOR_H
