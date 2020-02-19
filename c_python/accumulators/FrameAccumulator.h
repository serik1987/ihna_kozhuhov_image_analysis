//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FRAMEACCUMULATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FRAMEACCUMULATOR_H

extern "C" {

    typedef struct {
        PyImanA_AccumulatorObject parent;
    } PyImanA_FrameAccumulatorObject;

    static PyTypeObject PyImanA_FrameAccumulatorType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.accumulators.FrameAccumulator",
            /* tp_basicsize */ sizeof(PyImanA_FrameAccumulatorObject),
            /* tp_itemsize */ 0,
    };

    static PyObject* PyImanA_FrameAccumulator_GetPreprocessFilter(PyImanA_FrameAccumulatorObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* accumulator = (FrameAccumulator*)self->parent.handle;

        try{
            bool filter = accumulator->getPreprocessFilter();
            return PyBool_FromLong(filter);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanA_FrameAccumulator_SetPreprocessFilter
        (PyImanA_FrameAccumulatorObject* self, PyObject* value, void*){
        using namespace GLOBAL_NAMESPACE;

        bool preprocess_filter;
        if (!PyBool_Check(value)){
            PyErr_SetString(PyExc_ValueError, "preprocess_filter can be either True or False");
            return -1;
        }
        preprocess_filter = (value == Py_True);

        try{
            auto* accumulator = (FrameAccumulator*)self->parent.handle;
            accumulator->setPreprocessFilter(preprocess_filter);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanA_FrameAccumulator_GetPreprocessFilterRadius(PyImanA_FrameAccumulatorObject* self, void*){
        using namespace GLOBAL_NAMESPACE;

        try{
            auto* accumulator = (FrameAccumulator*)self->parent.handle;
            int radius = accumulator->getPreprocessFilterRadius();
            return PyLong_FromLong(radius);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanA_FrameAccumulator_SetPreprocessFilterRadius
            (PyImanA_FrameAccumulatorObject* self, PyObject* value, void*){
        using namespace GLOBAL_NAMESPACE;

        int radius;
        if (!PyLong_Check(value)){
            return -1;
        }
        radius = PyLong_AsLong(value);

        try{
            auto* accumulator = (FrameAccumulator*)self->parent.handle;
            accumulator->setPreprocessFilterRadius(radius);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanA_FrameAccumulator_GetDivideByAverage(PyImanA_FrameAccumulatorObject* self, void*){
        using namespace GLOBAL_NAMESPACE;

        try{
            auto* accumulator = (FrameAccumulator*)self->parent.handle;
            bool divide_by_average = accumulator->isDivideByAverage();
            return PyBool_FromLong(divide_by_average);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanA_FrameAccumulator_SetDivideByAverage
            (PyImanA_FrameAccumulatorObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;

        bool divide_by_average;
        if (!PyBool_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "Please, set True to enable this option or False to disable");
            return -1;
        }
        divide_by_average = (arg == Py_True);

        try{
            auto* accumulator = (FrameAccumulator*)self->parent.handle;
            accumulator->setDivideByAverage(divide_by_average);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyGetSetDef PyImanA_FrameAccumulator_Properties[] = {
            {(char*)"preprocess_filter", (getter)PyImanA_FrameAccumulator_GetPreprocessFilter,
                    (setter)PyImanA_FrameAccumulator_SetPreprocessFilter,
                    (char*)"True if low-pass spatial filter will be applied immediately after frame reading, before\n"
                           "isoline remove. False, if the filter shall be switched off"},

            {(char*)"preprocess_filter_radius", (getter)PyImanA_FrameAccumulator_GetPreprocessFilterRadius,
             (setter)PyImanA_FrameAccumulator_SetPreprocessFilterRadius,
             (char*)"Radius of the Low-pass filter, that will be applied immediately before frame reading. Reasonable\n"
                    "only if preprocess_filter is True, otherwise doesn't take place"},

            {(char*)"divide_by_average", (getter)PyImanA_FrameAccumulator_GetDivideByAverage,
             (setter)PyImanA_FrameAccumulator_SetDivideByAverage,
             (char*)"if this option is True, the average across all frames within the analysis epoch (before isoline \n"
                    "remove but after preprocess filter) will be calculated after analysis. The resultant map will \n"
                    "be divided by the average across all frames"},

            {NULL}
    };

    static int PyImanA_FrameAccumulator_Create(PyObject* module){
        PyImanA_FrameAccumulatorType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanA_FrameAccumulatorType.tp_doc =
                "This is the base class for all frame accumulators\n"
                "The frame accumulator accepts the native imaging record, synchronizes it, removes all isolines \n"
                "from it and then computes the data averaged across all frames\n"
                "This is an abstract class. Don't use it. Use any of its derivatives";
        PyImanA_FrameAccumulatorType.tp_base = &PyImanA_AccumulatorType;
        PyImanA_FrameAccumulatorType.tp_getset = PyImanA_FrameAccumulator_Properties;

        if (PyType_Ready(&PyImanA_FrameAccumulatorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanA_FrameAccumulatorType);
        PyImanA_FrameAccumulator_Handle = (PyObject*)&PyImanA_FrameAccumulatorType;

        if (PyModule_AddObject(module, "_accumulators_FrameAccumulator", PyImanA_FrameAccumulator_Handle) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAMEACCUMULATOR_H
