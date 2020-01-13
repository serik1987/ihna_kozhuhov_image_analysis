//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_LINEARFITISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_LINEARFITISOLINE_H

extern "C" {

    typedef struct {
        PyImanI_IsolineObject super;
    } PyImanI_LinearFitIsolineObject;

    static PyTypeObject PyImanI_LinearFitIsolineType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.isolines.LinearFitIsoline",
            .tp_basicsize = sizeof(PyImanI_LinearFitIsolineObject),
            .tp_itemsize = 0,
    };

    static int PyImanI_LinearFitIsoline_Init(PyImanI_LinearFitIsolineObject* self, PyObject* args, PyObject* kwds){
        if (PyImanI_Isoline_SetParent((PyImanI_IsolineObject*)self, args, kwds) < 0){
            return -1;
        }

        using namespace GLOBAL_NAMESPACE;
        try{
            auto* train_object = (PyImanS_StreamFileTrainObject*)self->super.parent_train;
            auto* sync_object = (PyImanY_SynchronizationObject*)self->super.parent_synchronization;
            auto* train = (StreamFileTrain*)train_object->super.train_handle;
            auto* sync = (Synchronization*)sync_object->synchronization_handle;
            auto* isoline = new LinearFitIsoline(*train, *sync);
            self->super.isoline_handle = isoline;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static int PyImanI_LinearFitIsoline_Create(PyObject* module){

        PyImanI_LinearFitIsolineType.tp_base = &PyImanI_IsolineType;
        PyImanI_LinearFitIsolineType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanI_LinearFitIsolineType.tp_doc =
                "Provides isoline estimation that is based on the linear regression of the whole signal in \n"
                "investigated interval\n";
        PyImanI_LinearFitIsolineType.tp_init = (initproc)PyImanI_LinearFitIsoline_Init;

        if (PyType_Ready(&PyImanI_LinearFitIsolineType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanI_LinearFitIsolineType);
        PyImanI_LinearFitIsoline_Handle = &PyImanI_LinearFitIsolineType;

        if (PyModule_AddObject(module, "_isolines_LinearFitIsoline", (PyObject*)&PyImanI_LinearFitIsolineType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_LINEARFITISOLINE_H
