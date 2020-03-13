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
//        std::exception e;
//        iman_exception e("Sample IMAN exception");
//        Accumulator::AccumulatorException e("Sample accumulator exception");
        FrameAccumulator::BadPreprocessFilterRadiusException e;
        PyIman_Exception_process(&e);
        return NULL;
    }
    #endif

    static PyObject* PyIman_Convolve(PyObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        using namespace std;
        PyArrayObject* input_data_object = NULL;
        npy_intp* dims;
        PyArrayObject* output_data_object = NULL;
        double dradius;
        int r;
        int* aindex_x = nullptr;
        int* aindex_y = nullptr;
        int aindex_n = 0;

        try{
            if (!PyArg_ParseTuple(args, "Od", &input_data_object, &dradius)){
                return NULL;
            }
            if (PyArray_NDIM(input_data_object) != 2){
                PyErr_SetString(PyExc_ValueError, "The data matrix shall have two dimensions");
            }
            dims = PyArray_DIMS(input_data_object);

            r = (int)floor(dradius);
            aindex_x = new int[(1+2*r) * (1+2*r)];
            aindex_y = new int[(1+2*r) * (1+2*r)];
            for (int j=-r; j<=r; j++){
                for (int k=-r; k<=r; k++){
                    if (hypot((double)j, (double)k) <= dradius){
                        aindex_x[aindex_n] = k;
                        aindex_y[aindex_n++] = j;
                    }
                }
            }

            output_data_object = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
            for (int y0=0; y0 < dims[0]; ++y0){
                for (int x0=0; x0 < dims[1]; ++x0){
                    auto* output_value = (double*)PyArray_GETPTR2(output_data_object, y0, x0);
                    double dumd = 0.0;
                    int m = 0;
                    for (int j=0; j < aindex_n; ++j){
                        int x = x0 + aindex_x[j];
                        int y = y0 + aindex_y[j];
                        if (x > 0 && y > 0 && x < dims[1] && y < dims[0]){
                            auto* input_value = (double*)PyArray_GETPTR2(input_data_object, y, x);
                            dumd += *input_value;
                            m++;
                        }
                    }
                    if (m == 0){
                        *output_value = 0.0;
                    } else {
                        *output_value = dumd / m;
                    }
                }
            }

            delete [] aindex_x;
            delete [] aindex_y;
            return (PyObject*)output_data_object;
        } catch (std::exception& e){
            delete [] aindex_x;
            delete [] aindex_y;
            Py_XDECREF(output_data_object);
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyMethodDef PyIman_Methods[] = {
    #ifdef C_EXCEPTION_TEST
            {"test_exception", PyIman_Test_exception, METH_NOARGS, ""},
    #endif
            {"convolve", (PyCFunction)PyIman_Convolve, METH_VARARGS, ""},
            {NULL}
    };

    static PyModuleDef PyIman_Description = {
            PyModuleDef_HEAD_INIT,
            /* m_name */ "ihna.kozhukhov._imageanalysis",
            /* m_doc */ "",
            /* m_size */ -1,
            /* m_methods */ PyIman_Methods
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

        if (PyImanT_Init(imageanalysis) < 0){
            PyImanC_Destroy();
            PyImanS_Destroy();
            Py_DECREF(PyIman_ImanError);
            Py_DECREF(imageanalysis);
        }

        if (PyImanY_Init(imageanalysis) < 0){
            PyImanT_Destroy();
            PyImanC_Destroy();
            PyImanS_Destroy();
            Py_DECREF(PyIman_ImanError);
            Py_DECREF(imageanalysis);
        }

        if (PyImanI_Init(imageanalysis) < 0){
            PyImanY_Destroy();
            PyImanT_Destroy();
            PyImanC_Destroy();
            PyImanS_Destroy();
            Py_DECREF(PyIman_ImanError);
            Py_DECREF(imageanalysis);
        }

        if (PyImanA_Init(imageanalysis) < 0){
            PyImanI_Destroy();
            PyImanY_Destroy();
            PyImanT_Destroy();
            PyImanC_Destroy();
            PyImanS_Destroy();
            Py_DECREF(PyIman_ImanError);
            Py_DECREF(imageanalysis);
        }

        return imageanalysis;
    }

}