//
// Created by serik1987 on 03.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FRAME_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FRAME_H

#include <sstream>

static PyImanS_FrameObject* PyImanS_Frame_New(PyTypeObject* type, PyObject* args, PyObject* kwds){
    PyImanS_FrameObject* frame;
    frame = (PyImanS_FrameObject*)type->tp_alloc(type, 0);
    if (frame != NULL){
        frame->parent_train = NULL;
        frame->frame_handle = NULL;
    }
    return frame;
}

static int PyImanS_Frame_Init(PyImanS_FrameObject* self, PyObject* args, PyObject* kwds){
    if (!PyArg_ParseTuple(args, "O!", &PyImanS_FileTrainType, &self->parent_train)) {
        return -1;
    }
    Py_INCREF(self->parent_train);
    return 0;
}

static void PyImanS_Frame_Destroy(PyImanS_FrameObject* self){
#ifdef DEBUG_DELETE_CHECK
    printf("SO Frame delete\n");
#endif
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    frame->iLock = false;
    Py_XDECREF(self->parent_train);
    Py_TYPE(self)->tp_free(self);
}

static PyObject* PyImanS_Frame_Print(PyImanS_FrameObject* self){
    using namespace GLOBAL_NAMESPACE;
    std::stringstream ss;
    auto* frame = (Frame*)self->frame_handle;

    try{
        ss << *frame << "\n";
    } catch (std::exception& e){
        PyIman_Exception_process(&e);
        return NULL;
    }

    return PyUnicode_FromString(ss.str().c_str());
}

static PyObject* PyImanS_Frame_GetNumber(PyImanS_FrameObject* self, void*){
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    int number;

    try{
        number = frame->getFrameNumber();
    } catch (std::exception& e){
        PyIman_Exception_process(&e);
        return NULL;
    }

    return PyLong_FromLong(number);
}

static PyObject* PyImanS_Frame_GetSequentialNumber(PyImanS_FrameObject* self, void*){
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    int sequential_number;

    try{
        sequential_number = frame->getFramChunk().getFrameSequentialNumber();
    } catch (std::exception& e){
        PyIman_Exception_process(&e);
        return NULL;
    }

    return PyLong_FromLong(sequential_number);
}

static PyObject* PyImanS_Frame_GetArrivalTime(PyImanS_FrameObject* self, void*){
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    double time;

    try{
        time = frame->getFramChunk().getTimeArrival();
    } catch (std::exception& e){
        PyIman_Exception_process(&e);
        return NULL;
    }

    return PyFloat_FromDouble(time);
}

static PyObject* PyImanS_Frame_GetTimeDelay(PyImanS_FrameObject* self, void*){
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    double time;

    try{
        time = 1e-3 * frame->getFramChunk().getTimeDelayUsec();
    } catch (std::exception& e){
        PyIman_Exception_process(&e);
        return NULL;
    }

    return PyFloat_FromDouble(time);
}

static PyObject* PyImanS_Frame_GetSynchronizationValues(PyImanS_FrameObject* self, void*){
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    auto* train_object = (PyImanS_FileTrainObject*)self->parent_train;
    auto* train = (FileTrain*)train_object->train_handle;
    PyObject* result = NULL;

    try{
        int chans = train->getSynchronizationChannelNumber();
        result = PyTuple_New(chans);
        if (result == NULL) return NULL;
        for (int chan = 0; chan < chans; ++chan){
            uint32_t value = frame->getCostChunk().getSynchChannel(chan);
            PyObject* valueObject = PyLong_FromUnsignedLong(value);
            if (valueObject == NULL){
                Py_XDECREF(result);
                return NULL;
            };
            if (PyTuple_SetItem(result, chan, valueObject) < 0){
                Py_XDECREF(result);
                Py_XDECREF(valueObject);
                return NULL;
            }
        }
    } catch (std::exception& e){
        PyIman_Exception_process(&e);
    }

    return result;
}

static PyObject* PyImanS_Frame_GetSynchronizationDelays(PyImanS_FrameObject* self, void*){
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    auto* train_object = (PyImanS_FileTrainObject*)self->parent_train;
    auto* train = (FileTrain*)train_object->train_handle;
    PyObject* result = NULL;

    try{
        int chans = train->getSynchronizationChannelNumber();
        result = PyTuple_New(chans);
        if (result == NULL) return NULL;
        for (int chan = 0; chan < chans; ++chan){
            uint32_t delay = frame->getCostChunk().getSynchChannelDelay(chan);
            PyObject* delay_object = PyLong_FromUnsignedLong(delay);
            if (delay_object == NULL){
                Py_DECREF(result);
                return NULL;
            }
            if (PyTuple_SetItem(result, chan, delay_object) < 0){
                Py_DECREF(delay_object);
                Py_DECREF(result);
                return NULL;
            }
        }
    } catch (std::exception& e){
        PyIman_Exception_process(&e);
        return NULL;
    }

    return result;
}

static PyObject* PyImanS_Frame_GetBody(PyImanS_FrameObject* self, void*){
    using namespace GLOBAL_NAMESPACE;
    auto* frame = (Frame*)self->frame_handle;
    auto* train_object = (PyImanS_FileTrainObject*)self->parent_train;
    auto* train = (FileTrain*)train_object->train_handle;
    npy_intp dims[2];
    dims[0] = train->getYSize();
    dims[1] = train->getXSize();
    auto* array = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_UINT16);
    if (array == NULL){
        return NULL;
    }

    int idx = 0;
    for (int i = 0; i < train->getYSize(); ++i){
        for (int j = 0; j < train->getXSize(); ++j){
            auto* pval = (uint16_t*)PyArray_GETPTR2(array, i, j);
            *pval = frame->getBody()[idx];
            ++idx;
        }
    }

    return (PyObject*)array;
}

static PyGetSetDef PyImanS_Frame_properties[] = {
        {(char*)"number", (getter)PyImanS_Frame_GetNumber, NULL,
                (char*)"Frame number, after temporal binning", NULL},
        {(char*)"sequential_number", (getter)PyImanS_Frame_GetSequentialNumber, NULL,
         (char*)"Frame number, before temporal binning", NULL},
        {(char*)"arrival_time", (getter)PyImanS_Frame_GetArrivalTime, NULL,
         (char*)"Frame arrival time, ms", NULL},
        {(char*)"delay", (getter)PyImanS_Frame_GetTimeDelay, NULL,
         (char*)"Time delay, ms", NULL},
        {(char*)"synch", (getter)PyImanS_Frame_GetSynchronizationValues, NULL,
         (char*)"Values from synchronization channels", NULL},
        {(char*)"synch_delay", (getter)PyImanS_Frame_GetSynchronizationDelays, NULL,
         (char*)"Synchronization channel delays", NULL},
        {(char*)"body", (getter)PyImanS_Frame_GetBody, NULL,
         (char*)"The frame data as numpy matrix", NULL},
        {NULL}
};

static int PyImanS_Frame_Create(PyObject* module){

    PyImanS_FrameType.tp_doc = "The object represents a single frame. The only way is to use subscript index for the \n"
                               "train";
    PyImanS_FrameType.tp_new = (newfunc)PyImanS_Frame_New;
    PyImanS_FrameType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyImanS_FrameType.tp_dealloc = (destructor)PyImanS_Frame_Destroy;
    PyImanS_FrameType.tp_init = (initproc)PyImanS_Frame_Init;
    PyImanS_FrameType.tp_str = (reprfunc)PyImanS_Frame_Print;
    PyImanS_FrameType.tp_getset = PyImanS_Frame_properties;

    if (PyType_Ready(&PyImanS_FrameType) < 0){
        return -1;
    }

    Py_INCREF(&PyImanS_FrameType);
    PyImanS_FrameType_Handle = (PyObject*)&PyImanS_FrameType;

    if (PyModule_AddObject(module, "_sourcefiles_Frame", PyImanS_FrameType_Handle) < 0){
        return -1;
    }

    return 0;
}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAME_H
