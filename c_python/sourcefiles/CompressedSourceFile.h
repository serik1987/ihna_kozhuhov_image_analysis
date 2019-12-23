//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSEDSOURCEFILE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_COMPRESSEDSOURCEFILE_H

extern "C" {

    typedef struct {
        PyImanS_TrainSourceFileObject super;
    } PyImanS_CompressedSourceFileObject;

    static PyTypeObject PyImanS_CompressedSourceFileType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._CompressedSourceFile",
            .tp_basicsize = sizeof(PyImanS_CompressedSourceFileObject),
            .tp_itemsize = 0,
    };

    static PyImanS_CompressedSourceFileObject* PyImanS_CompressedSourceFile_New
            (PyTypeObject* cls, PyObject* args, PyObject* kwds){
        auto* self = (PyImanS_CompressedSourceFileObject*)PyImanS_TrainSourceFile_New(cls, args, kwds);
        return self;
    }

    static int PyImanS_CompressedSourceFile_Init(PyImanS_CompressedSourceFileObject* self,
            PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        const char* path;
        const char* file;
        int traverse_mode_id;
        TrainSourceFile::NotInHead traverse_mode;
        PyObject* parent;

        if (PyImanS_TrainSourceFile_ParseArguments(args, &path, &file, &traverse_mode_id, &parent) < 0){
            return -1;
        }

        if (traverse_mode_id == PyImanS_TrainSourceFile_TraverseMode_Traverse){
            traverse_mode = TrainSourceFile::NotInHeadTraverse;
        } else if (traverse_mode_id == PyImanS_TrainSourceFile_TraverseMode_Exception){
            traverse_mode = TrainSourceFile::NotInHeadFail;
        } else if (traverse_mode_id == PyImanS_TrainSourceFile_TraverseMode_Ignore){
            traverse_mode = TrainSourceFile::NotInHeadIgnore;
        } else {
            PyErr_SetString(PyExc_ValueError, "Please, upgrade PyImanS_CompressedSourceFile_Init");
            Py_XDECREF(parent);
            return -1;
        }

        if (parent == NULL){
            self->super.parent_train = NULL;
            self->super.super.file_handle =
                    new CompressedSourceFile(path, file, traverse_mode);
        } else {
            self->super.parent_train = parent;
            self->super.super.file_handle = self;
        }

        return 0;
    }

    static int PyImanS_CompressedSourceFile_Create(PyObject* module){

        PyImanS_CompressedSourceFileType.tp_base = &PyImanS_TrainSourceFileType;
        PyImanS_CompressedSourceFileType.tp_doc = "Use CompressedSourceFile instead";
        PyImanS_CompressedSourceFileType.tp_new = (newfunc)PyImanS_CompressedSourceFile_New;
        PyImanS_CompressedSourceFileType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_CompressedSourceFileType.tp_init = (initproc)PyImanS_CompressedSourceFile_Init;

        if (PyType_Ready(&PyImanS_CompressedSourceFileType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_CompressedSourceFileType);

        if (PyModule_AddObject(module, "_sourcefiles_CompressedSourceFile",
                               (PyObject*)&PyImanS_CompressedSourceFileType) < 0){
            Py_DECREF(&PyImanS_CompressedSourceFileType);
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSEDSOURCEFILE_H
