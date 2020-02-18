//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ACCUMULATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ACCUMULATOR_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        PyObject* corresponding_isoline;
        void* handle;
    } PyImanA_AccumulatorObject;

    static PyTypeObject PyImanA_AccumulatorType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.accumulators.Accumulator",
            /* tp_basicsize */ sizeof(PyImanA_AccumulatorObject),
            /* tp_itemsize */ 0
    };

    static PyImanA_AccumulatorObject* PyImanA_Accumulator_New(PyTypeObject* cls, PyObject* args, PyObject* kwds){
        auto* self = (PyImanA_AccumulatorObject*)cls->tp_alloc(cls, 0);
        if (self != NULL){
            self->corresponding_isoline = NULL;
            self->handle = NULL;
        }
        return self;
    }

    static int PyImanA_Accumulator_ArgumentCheck(PyImanA_AccumulatorObject* acc, PyObject* arg){
        PyObject* isoline;

        if (!PyArg_ParseTuple(arg, "O!", &PyImanI_IsolineType, &isoline)){
            return -1;
        }

        Py_INCREF(isoline);
        acc->corresponding_isoline = isoline;

        return 0;
    }

    static int PyImanA_Accumulator_Init(PyImanA_AccumulatorObject* self, PyObject* arg, PyObject* kwds){
        PyErr_SetString(PyExc_NotImplementedError, "This is an abstract class. Use any of its derivatives");
        return -1;
    }

    static void PyImanA_Accumulator_Destroy(PyImanA_AccumulatorObject* self){
        using namespace GLOBAL_NAMESPACE;

        if (self->handle != NULL){
            auto* accumulator = (Accumulator*)self->handle;
            delete accumulator;
        }
        Py_XDECREF(self->corresponding_isoline);

        Py_TYPE(self)->tp_free(self);
    }

    static int PyImanA_Accumulator_Create(PyObject* module){

        PyImanA_AccumulatorType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanA_AccumulatorType.tp_doc =
                "This is the base class for all accumulators\n"
                "Accumulator is an object that loads native file train, synchronizes it, removes the isoline and\n"
                "then extracts the data averaged across time or across ROI\n"
                "This is the base class. Don't use. it. Use any derivative of the accumulator instead\n";
        PyImanA_AccumulatorType.tp_new = (newfunc)PyImanA_Accumulator_New;
        PyImanA_AccumulatorType.tp_dealloc = (destructor)PyImanA_Accumulator_Destroy;
        PyImanA_AccumulatorType.tp_init = (initproc)PyImanA_Accumulator_Init;

        if (PyType_Ready(&PyImanA_AccumulatorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanA_AccumulatorType);
        PyImanA_Accumulator_Handle = (PyObject*)&PyImanA_AccumulatorType;

        if (PyModule_AddObject(module, "_accumulators_Accumulator", PyImanA_Accumulator_Handle) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ACCUMULATOR_H
