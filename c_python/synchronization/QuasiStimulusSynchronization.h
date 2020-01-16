//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_QUASISTIMULUSSYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_QUASISTIMULUSSYNCHRONIZATION_H

extern "C" {

    typedef struct {
        PyImanY_SynchronizationObject super;
    } PyImanY_QuasiStimulusSynchronizationObject;

    static PyTypeObject PyImanY_QuasiStimulusSynchronizationType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.synchronization.QuasiStimulusSynchronization",
            .tp_basicsize = sizeof(PyImanY_QuasiStimulusSynchronizationObject),
            .tp_itemsize = 0,
    };

    static int PyImanY_QuasiStimulusSynchronization_Init
            (PyImanY_QuasiStimulusSynchronizationObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;

        if (PyImanY_Synchronization_SetParent((PyImanY_SynchronizationObject*)self, args) < 0){
            return -1;
        }

        try{
            auto* train_object = self->super.parent_train;
            auto* train = (StreamFileTrain*)train_object->super.train_handle;
            printf("SO passed\n");
            auto* sync = new QuasiStimulusSynchronization(*train);
            self->super.synchronization_handle = sync;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static int PyImanY_QuasiStimulusSynchronization_Create(PyObject* module){

        PyImanY_QuasiStimulusSynchronizationType.tp_base = &PyImanY_SynchronizationType;
        PyImanY_QuasiStimulusSynchronizationType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanY_QuasiStimulusSynchronizationType.tp_doc =
                "Performs the synchronization that based on assumption that stimulus period equals to \n"
                "integer number of timestamps. The class requires to set a stimulus period in timestamp number\n"
                "1) Stimulus phase at timestamps that are multiple to the stimulus period is multiple of 2*PI\n"
                "2) Initial and final frame are set in such a way as they contain integer number of stimulus periods\n"
                "\n"
                "Usage: sync = QuasiStimulusSynchronization(train)\n"
                "where: train is a certain train which is assumed to be opened";
        PyImanY_QuasiStimulusSynchronizationType.tp_init = (initproc)PyImanY_QuasiStimulusSynchronization_Init;

        if (PyType_Ready(&PyImanY_QuasiStimulusSynchronizationType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanY_QuasiStimulusSynchronizationType);
        PyImanY_QuasiStimulusSynchronization_Handle = (PyObject*)&PyImanY_QuasiStimulusSynchronizationType;

        if (PyModule_AddObject(module, "_synchronization_QuasiStimulusSynchronization",
                               (PyObject*)&PyImanY_QuasiStimulusSynchronizationType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H
