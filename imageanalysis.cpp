//
// Created by serik1987 on 21.12.2019.
//

// #define C_EXCEPTION_TEST
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "structmember.h"
#include <numpy/ndarrayobject.h>
#include "init.h"
#include "c_python/__init__.h"

extern "C" {

    #ifdef C_EXCEPTION_TEST
    static PyObject *PyIman_Test_exception(PyObject *, PyObject *) {
        using namespace GLOBAL_NAMESPACE;
        std::exception e;
//        iman_exception e("Sample exception message");
//        io_exception e("Sample IO exception", "FILE001.DAT");
//        FileTrain::train_exception e("Sample IO exception", "TRAIN001.DAT");
//        Decompressor::decompression_exception e;
        PyIman_Exception_process(&e);
        return NULL;
    }
    #endif

    static PyMethodDef PyIman_Methods[] = {
    #ifdef C_EXCEPTION_TEST
            {"test_exception", PyIman_Test_exception, METH_NOARGS, ""},
    #endif
            {NULL}
    };

    static PyModuleDef PyIman_Description = {
            PyModuleDef_HEAD_INIT,
            .m_name = "ihna.kozhukhov._imageanalysis",
            .m_doc = "",
            .m_size = -1,
            .m_methods = PyIman_Methods
    };

    PyMODINIT_FUNC PyInit__imageanalysis(void) {
        PyObject *imageanalysis;

        printf("(C) Valery Kalatsky, 2003\n");
        printf("When using this program reference to the following paper is mandatory:\n");
        printf("Kalatsky V.A., Stryker P.S. New Paradigm for Optical Imaging: Temporally\n");
        printf("Encoded Maps of Intrinsic Signal. Neuron. 2003. V. 38. N. 4. P. 529-545\n");
        printf("(C) Sergei Kozhukhov, 2020\n");
        printf("(C) the Institute of Higher Nervous Activity and Neurophysiology, \n");
        printf("Russian Academy of Sciences, 2020\n");

        imageanalysis = PyModule_Create(&PyIman_Description);
        if (imageanalysis == NULL) return NULL;
        import_array()

        PyIman_ImanError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.ImanError",
                                                     "This is the base exception that will be thrown by method within this package",
                                                     NULL, NULL);
        if (PyModule_AddObject(imageanalysis, "ImanError", PyIman_ImanError) < 0) {
            Py_DECREF(imageanalysis);
            return NULL;
        }

        if (PyImanS_Init(imageanalysis) < 0){
            Py_DECREF(PyIman_ImanError);
            Py_DECREF(imageanalysis);
            return NULL;
        }

        if (PyImanC_Init(imageanalysis) < 0){
            PyImanS_Destroy();
            Py_DECREF(PyIman_ImanError);
            Py_DECREF(imageanalysis);
        }

        return imageanalysis;
    }

}