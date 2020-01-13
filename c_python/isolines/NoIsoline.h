//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_NOISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_NOISOLINE_H

extern "C" {

    typedef struct {
        PyImanI_IsolineObject super;
    } PyImanI_NoIsolineObject;

    static PyTypeObject PyImanI_NoIsolineType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.isolines.NoIsoline",
            .tp_basicsize = sizeof(PyImanI_NoIsolineObject),
            .tp_itemsize = 0,
    };

    static int PyImanI_NoIsoline_Init(PyImanI_NoIsolineObject* self, PyObject* args, PyObject* kwds){
        if (PyImanI_Isoline_SetParent((PyImanI_IsolineObject*)self, args, kwds) < 0){
            return -1;
        }

        using namespace GLOBAL_NAMESPACE;
        try{
            auto* train_object = (PyImanS_StreamFileTrainObject*)self->super.parent_train;
            auto* train = (StreamFileTrain*)train_object->super.train_handle;
            auto* sync_object = (PyImanY_SynchronizationObject*)self->super.parent_synchronization;
            auto* sync = (Synchronization*)sync_object->synchronization_handle;
            auto* isoline = new NoIsoline(*train, *sync);
            self->super.isoline_handle = isoline;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static int PyImanI_NoIsoline_Create(PyObject* module){

        PyImanI_NoIsolineType.tp_base = &PyImanI_IsolineType;
        PyImanI_NoIsolineType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanI_NoIsolineType.tp_doc =
                "Use instance of this class if you don't want to remove any isoline";
        PyImanI_NoIsolineType.tp_init = (initproc)PyImanI_NoIsoline_Init;

        if (PyType_Ready(&PyImanI_NoIsolineType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanI_NoIsolineType);
        PyImanI_NoIsoline_Handle = &PyImanI_NoIsolineType;

        if (PyModule_AddObject(module, "_isolines_NoIsoline", (PyObject*)&PyImanI_NoIsolineType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_NOISOLINE_H
