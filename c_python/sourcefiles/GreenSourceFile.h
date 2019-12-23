//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_GREENSOURCEFILE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_GREENSOURCEFILE_H

extern "C" {

    typedef struct {
        PyImanS_SourceFileObject super;
    } PyImanS_GreenSourceFileObject;

    static PyImanS_GreenSourceFileObject* PyImanS_GreenSourceFile_New(PyTypeObject* cls, PyObject* args,
            PyObject* kwds){
        auto* self = PyImanS_SourceFile_New(cls, args, kwds);
        return (PyImanS_GreenSourceFileObject*)self;
    }

    static int PyImanS_GreenSourceFile_Init(PyImanS_GreenSourceFileObject* self, PyObject* args, PyObject*){
        const char* path;
        const char* file;

        if (!PyArg_ParseTuple(args, "ss", &path, &file)){
            return -1;
        }

        self->super.file_handle = new GLOBAL_NAMESPACE::GreenSourceFile(path, file);

        return 0;
    }

    static PyTypeObject PyImanS_GreenSourceFileType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles._GreenSourceFile",
            .tp_basicsize = sizeof(PyImanS_GreenSourceFileObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_GreenSourceFile_Create(PyObject* module){

        PyImanS_GreenSourceFileType.tp_base = &PyImanS_SourceFileType;
        PyImanS_GreenSourceFileType.tp_doc = "Use GreenSourceFile instead";
        PyImanS_GreenSourceFileType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_GreenSourceFileType.tp_new = (newfunc)PyImanS_GreenSourceFile_New;
        PyImanS_GreenSourceFileType.tp_init = (initproc)PyImanS_GreenSourceFile_Init;

        if (PyType_Ready(&PyImanS_GreenSourceFileType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_GreenSourceFileType);

        if (PyModule_AddObject(module, "_sourcefiles_GreenSourceFile", (PyObject*)&PyImanS_GreenSourceFileType) < 0){
            Py_DECREF(&PyImanS_GreenSourceFileType);
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_GREENSOURCEFILE_H
