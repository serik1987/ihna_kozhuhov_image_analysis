//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRAINSOURCEFILE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRAINSOURCEFILE_H

#define PyImanS_TrainSourceFile_TraverseMode_Traverse 0
#define PyImanS_TrainSourceFile_TraverseMode_Exception 1
#define PyImanS_TrainSourceFile_TraverseMode_Ignore 2

extern "C" {

    typedef struct {
        PyImanS_SourceFileObject super;
        PyObject* parent_train;
    } PyImanS_TrainSourceFileObject;

    static PyImanS_TrainSourceFileObject* PyImanS_TrainSourceFile_New(PyTypeObject* cls,
            PyObject* args, PyObject* kwds){
        auto* super_self = PyImanS_SourceFile_New(cls, args, kwds);
        PyImanS_TrainSourceFileObject* self = NULL;

        if (super_self != NULL){
            self = (PyImanS_TrainSourceFileObject*)super_self;
            self->parent_train = NULL;
        }

        return self;
    }

    static int PyImanS_TrainSourceFile_ParseArguments(PyObject* args, const char** ppath, const char** pfile,
                                                      int* ptraverse_mode, PyObject** pparent){
        const char* traverse_mode_string;

        if (!PyArg_ParseTuple(args, "sssO", ppath, pfile, &traverse_mode_string, pparent)){
            return -1;
        }

        if (strcmp(traverse_mode_string, "traverse") == 0){
            *ptraverse_mode = PyImanS_TrainSourceFile_TraverseMode_Traverse;
        } else if (strcmp(traverse_mode_string, "exception") == 0){
            *ptraverse_mode = PyImanS_TrainSourceFile_TraverseMode_Exception;
        } else if (strcmp(traverse_mode_string, "ignore") == 0){
            *ptraverse_mode = PyImanS_TrainSourceFile_TraverseMode_Ignore;
        } else {
            PyErr_SetString(PyExc_ValueError, "traverse mode shall be one of the following values: "
                                              "'traverse', 'exception', 'ignore'");
            return -1;
        }

        if (*pparent == Py_None){
            *pparent = NULL;
        }  else {
            Py_INCREF(*pparent);
        }

        return 0;
    }

    static int PyImanS_TrainSourceFile_Init(PyImanS_TrainSourceFileObject*, PyObject*, PyObject*){
        PyErr_SetString(PyExc_NotImplementedError, "TrainSourceFile class is purely abstract");

        return -1;
    }

    static PyTypeObject PyImanS_TrainSourceFileType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles.TrainSourceFile",
            .tp_basicsize = sizeof(PyImanS_TrainSourceFileObject),
            .tp_itemsize = 0,
    };

    static void PyImanS_TrainSourceFile_Destroy(PyImanS_TrainSourceFileObject* self){
        if (self->parent_train != NULL){
            self->super.file_handle = NULL;
            Py_DECREF(self->parent_train);
        }

        PyImanS_SourceFile_Destroy((PyImanS_SourceFileObject*)self);
    }

    static int PyImanS_TrainSourceFile_Create(PyObject* module){
        PyImanS_TrainSourceFileType.tp_base = &PyImanS_SourceFileType;
        PyImanS_TrainSourceFileType.tp_doc = "This is the base class for StreamSourceFile and CompressedSourceFile";
        PyImanS_TrainSourceFileType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_TrainSourceFileType.tp_new = (newfunc)PyImanS_TrainSourceFile_New;
        PyImanS_TrainSourceFileType.tp_init = (initproc)PyImanS_TrainSourceFile_Init;
        PyImanS_TrainSourceFileType.tp_dealloc = (destructor)PyImanS_TrainSourceFile_Destroy;

        if (PyType_Ready(&PyImanS_TrainSourceFileType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_TrainSourceFileType);

        if (PyModule_AddObject(module, "_sourcefiles_TrainSourceFile", (PyObject*)&PyImanS_TrainSourceFileType) < 0){
            Py_DECREF(&PyImanS_TrainSourceFileType);
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRAINSOURCEFILE_H
