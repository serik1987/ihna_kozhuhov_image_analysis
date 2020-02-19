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
        PyObject* progress_bar;
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
            self->progress_bar = NULL;
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
        Py_XDECREF(self->progress_bar);

        Py_TYPE(self)->tp_free(self);
    }

    static PyObject* PyImanA_Accumulator_GetChannelNumber(PyImanA_AccumulatorObject* self, void*){
        using namespace GLOBAL_NAMESPACE;

        try{
            auto* accumulator = (Accumulator*)self->handle;
            int chans = accumulator->getChannelNumber();
            return PyLong_FromLong(chans);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanA_Accumulator_IsAccumulated(PyImanA_AccumulatorObject* self, void*){
        using namespace GLOBAL_NAMESPACE;

        try{
            auto* accumulator = (Accumulator*)self->handle;
            bool result = accumulator->isAccumulated();
            return PyBool_FromLong(result);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanA_Accumulator_Accumulate(PyImanA_AccumulatorObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        try{
            auto* accumulator = (Accumulator*)self->handle;
            accumulator->accumulate();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanA_Accumulator_SetProgressBar(PyImanA_AccumulatorObject* self,
            PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        PyObject* progress_bar;
        if (!PyArg_ParseTuple(args, "O", &progress_bar)){
            return NULL;
        }

        PyObject* result = PyObject_CallMethod(progress_bar, "progress_function", "iis", 0, 100, "Accumulation");
        if (result == NULL){
            return NULL;
        }

        if (!PyBool_Check(result)){
            Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "progress_function shall return boolean value");
            return NULL;
        }
        Py_DECREF(result);

        Py_INCREF(progress_bar);
        self->progress_bar = progress_bar;

        try{
            auto* accumulator = (Accumulator*)self->handle;
            accumulator->setProgressFunction(PyIman_ReadingProgressFunction, progress_bar);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

        return Py_BuildValue("");
    }

    static PyObject* PyImanA_Accumulator_Print(PyImanA_AccumulatorObject* self){
        using namespace GLOBAL_NAMESPACE;
        using namespace std;
        auto* accumulator = (Accumulator*)self->handle;

        try{
            stringstream ss;
            ss << *accumulator << endl;
            return PyUnicode_FromString(ss.str().c_str());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyGetSetDef PyImanA_Accumulator_Properties[] = {
            {(char*)"channel_number", (getter)PyImanA_Accumulator_GetChannelNumber, NULL,
            (char*)"Number of pixels on the map that will be used for the accumulation process"},

            {(char*)"is_accumulated", (getter)PyImanA_Accumulator_IsAccumulated, NULL,
             (char*)"True if the accumulation process is completed, false otherwise"},

            {NULL}
    };

    static PyMethodDef PyImanA_Accumulator_Methods[] = {
            {"accumulate", (PyCFunction)PyImanA_Accumulator_Accumulate, METH_NOARGS,
                    (char*)"Launches the accumulation process"},

            {"set_progress_bar", (PyCFunction)PyImanA_Accumulator_SetProgressBar, METH_VARARGS,
             (char*)"Sets the progress bar\n"
                    "Progress bar depends on certain GUI you use. In any way, it shall have the following method:\n"
                    "progress_function(completed, total, message)\n"
                    "During the accumulation process this method will be called automatically in order to report\n"
                    "about current progress. The following parameters shall be passed:\n"
                    "completed - number of completed stages\n"
                    "total - total number of stages\n"
                    "msg - message that describes current processing status\n"
                    "The progress function shall return True if the user intend to continue the accumulation process\n"
                    "or False if the accumulation process shall be aborted immediately"},

            {NULL}
    };

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
        PyImanA_AccumulatorType.tp_getset = PyImanA_Accumulator_Properties;
        PyImanA_AccumulatorType.tp_methods = PyImanA_Accumulator_Methods;
        PyImanA_AccumulatorType.tp_str = (reprfunc)PyImanA_Accumulator_Print;

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
