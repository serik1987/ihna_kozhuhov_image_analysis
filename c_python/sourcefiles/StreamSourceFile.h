//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_STREAMSOURCEFILE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_STREAMSOURCEFILE_H

extern "C" {

    typedef struct {
        PyImanS_TrainSourceFileObject super;
    } PyImanS_StreamSourceFileObject;

    static PyImanS_StreamSourceFileObject* PyImanS_StreamSourceFile_New(PyTypeObject* cls,
            PyObject* args, PyObject* kwds){
        auto* self = (PyImanS_StreamSourceFileObject*)PyImanS_TrainSourceFile_New(cls, args, kwds);
        return self;
    }

    static int PyImanS_StreamSourceFile_Init(PyImanS_StreamSourceFileObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        const char* path;
        const char* filename;
        int traverse_mode_id;
        TrainSourceFile::NotInHead traverse_mode;
        PyObject* parent;

        if (PyImanS_TrainSourceFile_ParseArguments(args, &path, &filename, &traverse_mode_id, &parent) < 0){
            return -1;
        }
        if (traverse_mode_id == PyImanS_TrainSourceFile_TraverseMode_Traverse){
            traverse_mode = TrainSourceFile::NotInHeadTraverse;
        } else if (traverse_mode_id == PyImanS_TrainSourceFile_TraverseMode_Exception){
            traverse_mode = TrainSourceFile::NotInHeadFail;
        } else if (traverse_mode_id == PyImanS_TrainSourceFile_TraverseMode_Ignore){
            traverse_mode = TrainSourceFile::NotInHeadIgnore;
        } else {
            Py_XDECREF(parent);
            PyErr_SetString(PyExc_RuntimeError, "Unimproved PyImanS_StreamSourceFile_Init");
            return -1;
        }

        if (parent == NULL){
            self->super.parent_train = NULL;
            self->super.super.file_handle = new StreamSourceFile(path, filename, traverse_mode);
        } else {
            self->super.parent_train = parent;
            self->super.super.file_handle = self;
        }

        return 0;
    }

    static PyTypeObject PyImanS_StreamSourceFileType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            "ihna.kozhukhov.imageanalysis.sourcefiles._StreamSourceFile",
            sizeof(PyImanS_StreamSourceFileObject),
            0,
    };

    static int PyImanS_StreamSourceFile_Create(PyObject* module){
        PyImanS_StreamSourceFileType.tp_base = &PyImanS_TrainSourceFileType;
        PyImanS_StreamSourceFileType.tp_doc = "Use StreamSourceFile instead";
        PyImanS_StreamSourceFileType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_StreamSourceFileType.tp_new = (newfunc)PyImanS_StreamSourceFile_New;
        PyImanS_StreamSourceFileType.tp_init = (initproc)PyImanS_StreamSourceFile_Init;

        if (PyType_Ready(&PyImanS_StreamSourceFileType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_StreamSourceFileType);

        if (PyModule_AddObject(module, "_sourcefiles_StreamSourceFile", (PyObject*)&PyImanS_StreamSourceFileType) < 0){
            Py_DECREF(&PyImanS_StreamSourceFileType);
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_STREAMSOURCEFILE_H
