//
// Created by serik1987 on 08.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_DECOMPRESSOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_DECOMPRESSOR_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        PyObject* source_train;
        char output_path[512];
        void* decompressor_handle;
        PyObject* progress_bar;
    } PyImanC_DecompressorObject;

    static PyTypeObject PyImanC_DecompressorType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        /* tp_name */ "ihna.kozhukhov.imageanalysis.compression._Decompressor",
        /* tp_basicsize */ sizeof(PyImanC_DecompressorObject),
        /* tp_itemsize */ 0,
    };

    static PyImanC_DecompressorObject* PyImanC_Decompressor_New(PyTypeObject* type, PyObject*, PyObject*){
        PyImanC_DecompressorObject* self = NULL;
        self = (PyImanC_DecompressorObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->source_train = NULL;
            self->decompressor_handle = NULL;
            self->progress_bar = NULL;
        }
        return self;
    }

    static int PyImanC_Decompressor_Init(PyImanC_DecompressorObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        const char* output = NULL;

        if (!PyArg_ParseTuple(args, "O!s", (PyObject*)&PyImanS_CompressedFileTrainType,
                &self->source_train, &output)){
            return -1;
        }

        Py_INCREF(self->source_train);
        strcpy(self->output_path, output);

        try{
            auto* train_object = (PyImanS_CompressedFileTrainObject*)self->source_train;
            auto* train = (CompressedFileTrain*)train_object->super.train_handle;
            auto* decompressor = new Decompressor(*train, self->output_path);
            self->decompressor_handle = decompressor;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }

        return 0;
    }

    static void PyImanC_Decompressor_Destroy(PyImanC_DecompressorObject* self){
        using namespace GLOBAL_NAMESPACE;
        Py_XDECREF(self->source_train);
        printf("SO Source train destroyed\n");
        Py_XDECREF(self->progress_bar);
        printf("SO Progress bar destroyed\n");

        if (self->decompressor_handle != NULL){
            auto* decompressor = (Decompressor*)self->decompressor_handle;
            delete decompressor;
            printf("SO C++ Decompressor object destroyed\n");
        }

        Py_TYPE(self)->tp_free(self);
    }

    static PyObject* PyImanC_Decompressor_Run(PyImanC_DecompressorObject* self, PyObject*, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        try{
            auto* decompressor = (Decompressor*)self->decompressor_handle;
            decompressor->run();
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
        return Py_BuildValue("");
    }

    static void PyImanC_Decompressor_ProgressBar(float perc, void* handle){
        auto* decompressor_object = (PyImanC_DecompressorObject*)handle;
        PyObject* function = decompressor_object->progress_bar;
        PyObject* result = PyObject_CallFunction(function, "f", perc);
        if (result != NULL){
            Py_DECREF(result);
        } else {
            printf("Decompressor::progressBar: finished with exception\n");
        }
    }

    static PyObject* PyImanC_Decompressor_SetProgressBar(PyImanC_DecompressorObject* self, PyObject* f, PyObject*){
        using namespace GLOBAL_NAMESPACE;

        PyObject* result = PyObject_CallFunction(f, "f", 0.0);
        if (result == NULL) return NULL;
        Py_DECREF(result);

        try{
            Py_XDECREF(self->progress_bar);
            self->progress_bar = f;
            Py_INCREF(f);
            auto* decompressor = (Decompressor*)self->decompressor_handle;
            decompressor->setProgressFunction(PyImanC_Decompressor_ProgressBar, self);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

        return Py_BuildValue("");
    }

    static PyObject* PyImanC_Decompressor_GetFullOutputFile(PyImanC_DecompressorObject* self, PyObject*, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        auto* decompressor = (Decompressor*)self->decompressor_handle;
        return Py_BuildValue("s", decompressor->getFullOutputFile().c_str());
    }

    static PyMethodDef PyImanC_Decompressor_methods[] = {
            {"run", (PyCFunction)PyImanC_Decompressor_Run, METH_NOARGS, ""},
            {"set_progress_bar", (PyCFunction)PyImanC_Decompressor_SetProgressBar, METH_O, ""},
            {"get_full_output_file", (PyCFunction)PyImanC_Decompressor_GetFullOutputFile, METH_NOARGS, ""},
            {NULL}
    };

    static int PyImanC_Decompressor_Create(PyObject *module) {

        PyImanC_DecompressorType.tp_flags = Py_TPFLAGS_DEFAULT;
        PyImanC_DecompressorType.tp_doc = "Use decompress(...) function instead";
        PyImanC_DecompressorType.tp_new = (newfunc)PyImanC_Decompressor_New;
        PyImanC_DecompressorType.tp_dealloc = (destructor)PyImanC_Decompressor_Destroy;
        PyImanC_DecompressorType.tp_init = (initproc)PyImanC_Decompressor_Init;
        PyImanC_DecompressorType.tp_methods = PyImanC_Decompressor_methods;

        if (PyType_Ready(&PyImanC_DecompressorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanC_DecompressorType);
        PyImanC_Class_list[PyImanC_Current_class++] = (PyObject*)&PyImanC_DecompressorType;

        if (PyModule_AddObject(module, "_compression_Decompressor", (PyObject*)&PyImanC_DecompressorType) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_DECOMPRESSOR_H
