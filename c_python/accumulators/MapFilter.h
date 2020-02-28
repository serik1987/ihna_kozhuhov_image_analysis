//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_MAPFILTER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_MAPFILTER_H

extern "C"{

    typedef struct {
        PyImanA_FrameAccumulatorObject parent;
    } PyImanA_MapFilterObject;

    static PyTypeObject PyImanA_MapFilterType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.accumulators.MapFilter",
            /* tp_basicsize */ sizeof(PyImanA_MapFilterObject),
            /* tp_itemsize */ 0
    };

    static int PyImanA_MapFilter_Init(PyImanA_MapFilterObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        if (PyImanA_Accumulator_ArgumentCheck((PyImanA_AccumulatorObject*)self, args) < 0){
            return -1;
        }

        try{
            auto* isoline_object = (PyImanI_IsolineObject*)self->parent.parent.corresponding_isoline;
            auto* isoline = (Isoline*)isoline_object->isoline_handle;
            auto* filter = new MapFilter(*isoline);
            self->parent.parent.handle = filter;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }

        return 0;
    }

    static PyObject* PyImanA_MapFilter_SetFilterCoefficients(
            PyImanA_MapFilterObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        PyObject* b_object;
        PyObject* a_object;
        auto* filter = (MapFilter*)self->parent.parent.handle;

        if (!PyArg_ParseTuple(args, "OO", &b_object, &a_object)){
            return NULL;
        }

        try{
            int b_length = PyObject_Length(b_object);
            int a_length = PyObject_Length(a_object);
            if (b_length == -1 || a_length == -1){
                return NULL;
            }
            std::vector<double> b(b_length);
            std::vector<double> a(a_length);
            for (int i=0; i < b_length;++i){
                PyObject* key = PyLong_FromLong(i);
                PyObject* item = PyObject_GetItem(b_object, key);
                Py_DECREF(key);
                if (item == NULL){
                    return NULL;
                }
                if (!PyFloat_Check(item)){
                    Py_DECREF(item);
                    return NULL;
                }
                b[i] = PyFloat_AsDouble(item);
                Py_DECREF(item);
            }
            for (int i=0; i < a_length; ++i){
                PyObject* key = PyLong_FromLong(i);
                PyObject* item = PyObject_GetItem(a_object, key);
                Py_DECREF(key);
                if (item == NULL){
                    return NULL;
                }
                if (!PyFloat_Check(item)){
                    Py_DECREF(item);
                    return NULL;
                }
                a[i] = PyFloat_AsDouble(item);
                Py_DECREF(item);
            }
            filter->setB(b);
            filter->setA(a);
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanA_MapFilter_GetTarget(PyImanA_MapFilterObject* self, void*){
        using namespace GLOBAL_NAMESPACE;

        try{
            auto* filter = (MapFilter*)self->parent.parent.handle;
            auto& train = filter->getTrain();
            const double* target = filter->getTargetMap();
            int y = train.getYSize();
            int x = train.getXSize();
            npy_intp dims[] = {y, x};
            PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
            for (int i=0; i < y; ++i){
                for (int j=0; j < x; ++j){
                    *(double*)PyArray_GETPTR2((PyArrayObject*)result, i, j) = *(target++);
                }
            }
            return result;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyMethodDef PyImanA_MapFilter_Methods[] = {
            {"set_filter", (PyCFunction)PyImanA_MapFilter_SetFilterCoefficients, METH_VARARGS,
             "Sets the filter coefficients\n"
             "\n"
             "Usage: f.set_filter(b, a)\n"
             "where b and a are nominator and denominator filter polynomials respectively\n"
             "See help on scipy.signal for detailed information about b and a polymials\n"
             "Also, use functions from this package to generate them"},

            {NULL}
    };

    static PyGetSetDef PyImanA_MapFilter_Properties[] = {
            {(char*)"target", (getter)PyImanA_MapFilter_GetTarget, NULL,
                    (char*)"The resultant map that represents distribution of the power of the filtered signal"},

            {NULL}
    };


    static int PyImanA_MapFilter_Create(PyObject* module){

        PyImanA_MapFilterType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanA_MapFilterType.tp_base = &PyImanA_FrameAccumulatorType;
        PyImanA_MapFilterType.tp_doc =
                "This object applies temporal filter to the imaging signal (w/o isoline) and then computes the signal\n"
                "squared-average across all frames.\n"
                "\n"
                "Usage: filter = MapFilter(isoline)\n"
                "where isoline is an object that removes an isoline (see ihna.kozhukhov.imageanalysis.isolines.Isoline\n"
                "for more details)";
        PyImanA_MapFilterType.tp_init = (initproc)PyImanA_MapFilter_Init;
        PyImanA_MapFilterType.tp_methods = PyImanA_MapFilter_Methods;
        PyImanA_MapFilterType.tp_getset = PyImanA_MapFilter_Properties;

        if (PyType_Ready(&PyImanA_MapFilterType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanA_MapFilterType);
        PyImanA_MapFilter_Handle = (PyObject*)&PyImanA_MapFilterType;

        if (PyModule_AddObject(module, "_accumulators_MapFilter", PyImanA_MapFilter_Handle) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPFILTER_H
