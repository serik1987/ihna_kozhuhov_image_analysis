//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_EXTERNALSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_EXTERNALSYNCHRONIZATION_H

extern "C" {

    typedef struct {
        PyImanY_SynchronizationObject super;
    } PyImanY_ExternalSynchronizationObject;

    static PyTypeObject PyImanY_ExternalSynchronizationType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.synchronization.ExternalSynchronization",
            .tp_basicsize = sizeof(PyImanY_ExternalSynchronizationObject),
            .tp_itemsize = 0,
    };

    static int PyImanT_ExternalSynchronization_Init(PyImanY_ExternalSynchronizationObject* self,
            PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        if (PyImanY_Synchronization_SetParent((PyImanY_SynchronizationObject*)self, args) < 0){
            return -1;
        }

        try{
            auto* train_object = self->super.parent_train;
            auto* train = (StreamFileTrain*)train_object->super.train_handle;
            auto* sync = new ExternalSynchronization(*train);
            self->super.synchronization_handle = sync;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static int PyImanY_ExternalSynchronization_Create(PyObject* module){

        PyImanY_ExternalSynchronizationType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanY_ExternalSynchronizationType.tp_base = &PyImanY_SynchronizationType;
        PyImanY_ExternalSynchronizationType.tp_doc =
                "Provides the synchronization based on the record from synchronization channel\n"
                "During the experiment the local phase of continuous visual stimulus is transmitted from \n"
                "the visual stimulator to the data acquisition setup\n"
                "In the recorded data such a signal is stored in so called 'synchronization channel'\n"
                "This object will perform the synchronization based on data from this channel\n"
                "1) Signal from the synchronization channel will be transformed to the timestamp-dependent\n"
                "stimulus phase\n"
                "2) The analysis epoch will be selected in such a way as it starts at the start of the stimulation\n"
                "cycle and finishes at the stimulation cycle. What cycle starts/finishes the stimulation is defined\n"
                "by the user\n"
                "3) the reference sin/cos will be estimated based on timestamp-dependent stimulus phase\n"
                "\n"
                "Usage: sync = ExternalSynchronization(train)\n"
                "train is an instance of StreamFileTrain already opened";
        PyImanY_ExternalSynchronizationType.tp_init = (initproc)PyImanT_ExternalSynchronization_Init;

        if (PyType_Ready(&PyImanY_ExternalSynchronizationType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanY_ExternalSynchronizationType);
        PyImanY_ExternalSynchronization_Handle = (PyObject*)&PyImanY_ExternalSynchronizationType;

        if (PyModule_AddObject(module, "_synchronization_ExternalSynchronization",
                               (PyObject*)&PyImanY_ExternalSynchronizationType) < 0){
            return -1;
        }


        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H
