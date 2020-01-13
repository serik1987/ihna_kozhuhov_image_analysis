//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TIMEAVERAGEISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TIMEAVERAGEISOLINE_H

extern "C" {

    typedef struct {
        PyImanI_IsolineObject super;
    } PyImanI_TimeAverageIsolineObject;

    static PyTypeObject PyImanI_TimeAverageIsolineType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.isolines.TimeAverageIsoline",
            .tp_basicsize = sizeof(PyImanI_TimeAverageIsolineObject),
            .tp_itemsize = 0,
    };

    static int PyImanI_TimeAverageIsoline_Init(PyImanI_TimeAverageIsolineObject* self,
        PyObject* args, PyObject* kwds){
        if (PyImanI_Isoline_SetParent((PyImanI_IsolineObject*)self, args, kwds) < 0){
            return -1;
        }

        using namespace GLOBAL_NAMESPACE;
        try{
            auto* train_object = (PyImanS_StreamFileTrainObject*)self->super.parent_train;
            auto* sync_object = (PyImanY_SynchronizationObject*)self->super.parent_synchronization;
            auto* train = (StreamFileTrain*)train_object->super.train_handle;
            auto* sync = (Synchronization*)sync_object->synchronization_handle;
            auto* isoline = new TimeAverageIsoline(*train, *sync);
            self->super.isoline_handle = isoline;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static int PyImanI_TimeAverageIsoline_Create(PyObject* module){

        PyImanI_TimeAverageIsolineType.tp_base = &PyImanI_IsolineType;
        PyImanI_TimeAverageIsolineType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanI_TimeAverageIsolineType.tp_doc =
                "The isoline will be calculated by smoothing the signal. This means that the isoline value at \n"
                "timestamp x is defined as signal average at the interval [x - N*C, x + N*C] where: \n"
                "C is a length of a single cycle in timestamps and N is some user-defined natural number\n";
        PyImanI_TimeAverageIsolineType.tp_init = (initproc)PyImanI_TimeAverageIsoline_Init;

        if (PyType_Ready(&PyImanI_TimeAverageIsolineType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanI_TimeAverageIsolineType);
        PyImanI_TimeAverageIsoline_Handle = &PyImanI_TimeAverageIsolineType;

        if (PyModule_AddObject(module, "_isolines_TimeAverageIsoline",
                               (PyObject*)&PyImanI_TimeAverageIsolineType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TIMEAVERAGEISOLINE_H
