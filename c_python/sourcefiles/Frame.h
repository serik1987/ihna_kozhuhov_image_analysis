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
    using namespace GLOBAL_NAMESPACE;
    Py_XDECREF(self->parent_train);
    auto* frame = (Frame*)self->frame_handle;
    frame->iLock = false;
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

static int PyImanS_Frame_Create(PyObject* module){

    PyImanS_FrameType.tp_doc = "The object represents a single frame. The only way is to use subscript index for the \n"
                               "train";
    PyImanS_FrameType.tp_new = (newfunc)PyImanS_Frame_New;
    PyImanS_FrameType.tp_flags = Py_TPFLAGS_DEFAULT;
    PyImanS_FrameType.tp_dealloc = (destructor)PyImanS_Frame_Destroy;
    PyImanS_FrameType.tp_init = (initproc)PyImanS_Frame_Init;
    PyImanS_FrameType.tp_str = (reprfunc)PyImanS_Frame_Print;

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
