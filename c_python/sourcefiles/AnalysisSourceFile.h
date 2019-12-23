//
// Created by serik1987 on 23.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_ANALYSISSOURCEFILE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_ANALYSISSOURCEFILE_H

extern "C" {

    typedef struct {
        PyImanS_SourceFileObject super;
    } PyImanS_AnalysisSourceFileObject;

    static PyImanS_AnalysisSourceFileObject* PyImanS_AnalysisSourceFile_New(PyTypeObject* cls,
            PyObject* args, PyObject* kwds){
        auto* self = PyImanS_SourceFile_New(cls, args, kwds);
        return (PyImanS_AnalysisSourceFileObject*)self;
    }

    static int PyImanS_AnalysisSourceFile_Init(PyImanS_AnalysisSourceFileObject* self, PyObject* args, PyObject*){
        const char* path = NULL;
        const char* file = NULL;

        if (!PyArg_ParseTuple(args, "ss", &path, &file)){
            return -1;
        }

        self->super.file_handle = new GLOBAL_NAMESPACE::AnalysisSourceFile(path, file);

        return 0;
    }

    static PyTypeObject PyImanS_AnalysisSourceFileType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imaging.sourcefiles._AnalysisSourceFile",
            .tp_basicsize = sizeof(PyImanS_AnalysisSourceFileObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_AnalysisSourceFile_Create(PyObject* module){

        PyImanS_AnalysisSourceFileType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_AnalysisSourceFileType.tp_base = &PyImanS_SourceFileType;
        PyImanS_AnalysisSourceFileType.tp_doc = "Use AnalysisSourceFile instead";
        PyImanS_AnalysisSourceFileType.tp_new = (newfunc)PyImanS_AnalysisSourceFile_New;
        PyImanS_AnalysisSourceFileType.tp_init = (initproc)PyImanS_AnalysisSourceFile_Init;

        if (PyType_Ready(&PyImanS_AnalysisSourceFileType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_AnalysisSourceFileType);

        if (PyModule_AddObject(module, "_sourcefiles_AnalysisSourceFile",
                (PyObject*)&PyImanS_AnalysisSourceFileType) < 0){
            Py_DECREF(&PyImanS_AnalysisSourceFileType);
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ANALYSISSOURCEFILE_H
