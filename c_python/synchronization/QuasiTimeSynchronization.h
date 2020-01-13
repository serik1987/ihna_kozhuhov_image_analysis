//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_QUASITIMESYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_QUASITIMESYNCHRONIZATION_H

extern "C" {

    typedef struct {
        PyImanY_SynchronizationObject super;
    } PyImanY_QuasiTimeSynchronizationObject;

    static PyTypeObject PyImanY_QuasiTimeSynchronizationType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.synchronization.QuasiTimeSynchronization",
            .tp_basicsize = sizeof(PyImanY_QuasiTimeSynchronizationObject),
            .tp_itemsize = 0,
    };

    static int PyImanY_QuasiTimeSynchronization_Init(PyImanY_QuasiTimeSynchronizationObject* self,
            PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        if (PyImanY_Synchronization_SetParent((PyImanY_SynchronizationObject*)self, args) < 0){
            return -1;
        }

        try{
            auto* train_object = self->super.parent_train;
            auto* train = (StreamFileTrain*)train_object->super.train_handle;
            auto* sync = new QuasiTimeSynchronization(*train);
            self->super.synchronization_handle = sync;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static int PyImanY_QuasiTimeSynchronization_Create(PyObject* module){

        PyImanY_QuasiTimeSynchronizationType.tp_base = &PyImanY_SynchronizationType;
        PyImanY_QuasiTimeSynchronizationType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanY_QuasiTimeSynchronizationType.tp_doc =
                "Provides a synchronization that is based on assumption the the stimulus has a definite period\n"
                "expressed in seconds, the frame arrival time ('TIME' channel) represents an exact time and the\n"
                "stimulus starts at the beginning of the record\n"
                "This means that: \n"
                "The reference signal will be plotted based on such assumption as its value at timestamps which \n"
                "frame arrival time is close to the stimulus period multiplied by N is 2 * PI * M\n"
                "Initial and final frame will be chosen in such a way as to contain integer number of cycles as this \n"
                "defined by the user.";
        PyImanY_QuasiTimeSynchronizationType.tp_init = (initproc)PyImanY_QuasiTimeSynchronization_Init;

        if (PyType_Ready(&PyImanY_QuasiTimeSynchronizationType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanY_QuasiTimeSynchronizationType);
        PyImanY_QuasiTimeSynchronization_Handle = (PyObject*)&PyImanY_QuasiTimeSynchronizationType;

        if (PyModule_AddObject(module, "_synchronization_QuasiTimeSynchronization",
                               (PyObject*)&PyImanY_QuasiTimeSynchronizationType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASITIMESYNCHRONIZATION_H
