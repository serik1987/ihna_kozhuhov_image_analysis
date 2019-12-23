//
// Created by serik1987 on 22.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_COMPRESSEDFILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_COMPRESSEDFILETRAIN_H

extern "C" {

    typedef struct {
        PyImanS_FileTrainObject super;
    } PyImanS_CompressedFileTrainObject;

    static PyImanS_CompressedFileTrainObject* PyImanS_CompressedFileTrain_New(PyTypeObject* cls,
            PyObject* args, PyObject* kwds){
        PyImanS_CompressedFileTrainObject* self = NULL;
        self = (PyImanS_CompressedFileTrainObject*)PyImanS_FileTrain_New(cls, args, kwds);
        return self;
    }

    static int PyImanS_CompressedFileTrain_Init(PyImanS_CompressedFileTrainObject* self,
            PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        const char* pathname = NULL;
        const char* filename = NULL;
        const char* traverse_mode_string = NULL;
        bool traverse_mode = false;

        if (!PyArg_ParseTuple(args, "sss", &pathname, &filename, &traverse_mode_string)){
            return -1;
        }

        if (strcmp(traverse_mode_string, "traverse") == 0){
            traverse_mode = true;
        } else if (strcmp(traverse_mode_string, "exception") == 0){
            traverse_mode = false;
        } else {
            PyErr_SetString(PyExc_ValueError, "Traverse mode may be either 'traverse' or 'exception'");
            return -1;
        }

        self->super.train_handle = new CompressedFileTrain(pathname, filename, traverse_mode);

        return 0;
    }

    static void PyImanS_CompressedFileTrain_Destroy(PyImanS_CompressedFileTrainObject* self){
        PyImanS_FileTrain_Destroy((PyImanS_FileTrainObject*)self);
    }

    static PyTypeObject PyImanS_CompressedFileTrainType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._CompressedFileTrain",
        .tp_basicsize = sizeof(PyImanS_CompressedFileTrainObject),
        .tp_itemsize = 0,
    };

    static int PyImanS_CompressedFileTrain_Create(PyObject* module){

        PyImanS_CompressedFileTrainType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_CompressedFileTrainType.tp_doc = "Use CompressedFileTrain instead";
        PyImanS_CompressedFileTrainType.tp_new = (newfunc)PyImanS_CompressedFileTrain_New;
        PyImanS_CompressedFileTrainType.tp_init = (initproc)PyImanS_CompressedFileTrain_Init;
        PyImanS_CompressedFileTrainType.tp_dealloc = (destructor)PyImanS_CompressedFileTrain_Destroy;
        PyImanS_CompressedFileTrainType.tp_base = &PyImanS_FileTrainType;

        if (PyType_Ready(&PyImanS_CompressedFileTrainType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_CompressedFileTrainType);

        if (PyModule_AddObject(module, "_sourcefiles_CompressedFileTrain",
                (PyObject*)&PyImanS_CompressedFileTrainType) < 0){
            Py_DECREF(&PyImanS_CompressedFileTrainType);
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDFILETRAIN_H
