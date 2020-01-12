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
            .tp_name = "ihna.kozhukhov.imageanalysis.synchronization.NoSynchronization",
            .tp_basicsize = sizeof(PyImanY_NoSynchronizationObject),
            .tp_itemsize = 0,
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

    static int PyImanY_NoSynchronization_Create(PyObject* module){
        printf("Creating NoSynchronization...\n");

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

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOSYNCHRONIZATION_H
