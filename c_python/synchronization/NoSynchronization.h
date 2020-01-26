//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_NOSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_NOSYNCHRONIZATION_H

extern "C" {

    typedef struct {
        PyImanY_SynchronizationObject super;
    } PyImanY_NoSynchronizationObject;

    static PyTypeObject PyImanY_NoSynchronizationType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.synchronization.NoSynchronization",
            /* tp_basicsize */ sizeof(PyImanY_NoSynchronizationObject),
            /* tp_itemsize */ 0,
    };

    static int PyImanY_NoSynchronization_Init(PyImanY_NoSynchronizationObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;

        if (PyImanY_Synchronization_SetParent((PyImanY_SynchronizationObject*)self, args) < 0){
            return -1;
        }

        try{
            PyImanS_StreamFileTrainObject* train_object = self->super.parent_train;
            auto* train = (StreamFileTrain*)train_object->super.train_handle;
            auto* sync = new NoSynchronization(*train);
            self->super.synchronization_handle = sync;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_NoSynchronization_SetInitialFrame(PyImanY_NoSynchronizationObject* self,
            PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (NoSynchronization*)self->super.synchronization_handle;
        int value;

        if (!PyArg_ParseTuple(args, "i", &value)){
            return NULL;
        }

        try{
            sync->setInitialFrame(value);
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_NoSynchronization_SetFinalFrame(PyImanY_NoSynchronizationObject* self,
        PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (NoSynchronization*)self->super.synchronization_handle;
        int value;

        if (!PyArg_ParseTuple(args, "i", &value)){
            return NULL;
        }

        try{
            sync->setFinalFrame(value);
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyMethodDef PyImanY_NoSynchronization_Methods[] = {
            {"set_initial_frame", (PyCFunction)PyImanY_NoSynchronization_SetInitialFrame, METH_VARARGS,
             "Sets the frame that starts an analysis"},

            {"set_final_frame", (PyCFunction)PyImanY_NoSynchronization_SetFinalFrame, METH_VARARGS,
             "Sets the frame that finishes an analysis"},

            {NULL}
    };

    static int PyImanY_NoSynchronization_Create(PyObject* module){

        PyImanY_NoSynchronizationType.tp_base = &PyImanY_SynchronizationType;
        PyImanY_NoSynchronizationType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanY_NoSynchronizationType.tp_doc =
                "This type of synchronization will not look for any stimulus\n"
                "This is suitable when you analyze the background activity or some stimulus-independent fratures\n"
                "This means that:\n"
                "1) The reference signal is 2PI multiplied by the timestamp value\n"
                "2) Initial and final frame is fully defined by the user\n"
                "3) At any harmonic value the program will simply sum all frames within the investigated range"
                "\n"
                "Usage: sync = ExternalSynchronization(train)\n"
                "train is an instance of StreamFileTrain already opened";
        PyImanY_NoSynchronizationType.tp_init = (initproc)PyImanY_NoSynchronization_Init;
        PyImanY_NoSynchronizationType.tp_methods = PyImanY_NoSynchronization_Methods;

        if (PyType_Ready(&PyImanY_NoSynchronizationType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanY_NoSynchronizationType);
        PyImanY_NoSynchronization_Handle = (PyObject*)&PyImanY_NoSynchronizationType;

        if (PyModule_AddObject(module, "_synchronization_NoSynchronization",
                (PyObject*)&PyImanY_NoSynchronizationType) < 0)
            return -1;

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOSYNCHRONIZATION_H
