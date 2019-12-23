//
// Created by serik1987 on 22.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_STREAMFILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_STREAMFILETRAIN_H

extern "C" {

    typedef struct {
        PyImanS_FileTrainObject super;
        uint32_t* sizes;
    } PyImanS_StreamFileTrainObject;

    static PyImanS_StreamFileTrainObject* PyImanS_StreamFileTrain_New(PyTypeObject* cls, PyObject* args, PyObject* kwds){
        PyImanS_StreamFileTrainObject* self;
        self = (PyImanS_StreamFileTrainObject*)PyImanS_FileTrain_New(cls, args, kwds);
        if (self != NULL){
            self->sizes = NULL;
        }
        return self;
    }

    static int PyImanS_StreamFileTrain_Init(PyImanS_StreamFileTrainObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        const char* pathname = NULL;
        const char* filename = NULL;
        PyObject* sizes_list = NULL;
        Py_ssize_t file_number = 0;
        const char* traverse_mode_string = NULL;
        bool traverse_mode = false;

        if (!PyArg_ParseTuple(args, "ssOs", &pathname, &filename, &sizes_list, &traverse_mode_string)){
            return -1;
        }

        if (!PyList_Check(sizes_list)){
            PyErr_SetString(PyExc_TypeError, "The third argument in the constructor shall be list of integers");
            return -1;
        }
        file_number = PyList_Size(sizes_list);
        self->sizes = new uint32_t[file_number];
        for (int i=0; i < file_number; ++i){
            PyObject* item = PyList_GetItem(sizes_list, i);
            if (!PyLong_Check(item)){
                PyErr_SetString(PyExc_TypeError, "The third argument shall be a list containing integer values only");
                return -1;
            }
            self->sizes[i] = PyLong_AsUnsignedLong(item);
            if (PyErr_Occurred()){
                return -1;
            }
        }

        if (strcmp(traverse_mode_string, "traverse") == 0){
            traverse_mode = true;
        } else if (strcmp(traverse_mode_string, "exception") == 0){
            traverse_mode = false;
        } else {
            PyErr_SetString(PyExc_ValueError, "Traverse mode may be either 'traverse' or 'exception'");
            return -1;
        }

        self->super.train_handle = new StreamFileTrain(pathname, filename, self->sizes, traverse_mode);

        return 0;
    }

    static void PyImanS_StreamFileTrain_Destroy(PyImanS_StreamFileTrainObject* self){
        if (self->sizes != NULL){
#ifdef DEBUG_DELETE_CHECK
            printf("FILE SIZES DELETE\n");
#endif
            delete [] self->sizes;
            self->sizes = NULL;
        }
        PyImanS_FileTrain_Destroy((PyImanS_FileTrainObject*)self);
    }

    static PyTypeObject PyImanS_StreamFileTrainType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis._StreamFileTrain",
            .tp_basicsize = sizeof(PyImanS_StreamFileTrainObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_StreamFileTrain_Create(PyObject* module){

        PyImanS_StreamFileTrainType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_StreamFileTrainType.tp_base = &PyImanS_FileTrainType;
        PyImanS_StreamFileTrainType.tp_doc = "Don't use this class directly. Use StreamFileTrain instead";
        PyImanS_StreamFileTrainType.tp_new = (newfunc)PyImanS_StreamFileTrain_New;
        PyImanS_StreamFileTrainType.tp_dealloc = (destructor)PyImanS_StreamFileTrain_Destroy;
        PyImanS_StreamFileTrainType.tp_init = (initproc)PyImanS_StreamFileTrain_Init;

        if (PyType_Ready(&PyImanS_StreamFileTrainType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_StreamFileTrainType);

        if (PyModule_AddObject(module, "_sourcefiles_StreamFileTrain", (PyObject*)&PyImanS_StreamFileTrainType) < 0){
            Py_DECREF(&PyImanS_StreamFileTrainType);
            return -1;
        }

        return 0;
    }

};


#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMFILETRAIN_H
