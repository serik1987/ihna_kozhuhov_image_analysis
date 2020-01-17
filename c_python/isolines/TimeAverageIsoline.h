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

    static PyObject* PyImanI_TimeAverageIsoline_GetAverageCycles(PyImanI_TimeAverageIsolineObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* isoline = (TimeAverageIsoline*)self->super.isoline_handle;

        try{
            return PyLong_FromLong(isoline->getAverageCycles());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanI_TimeAverageIsoline_SetAverageCycles
        (PyImanI_TimeAverageIsolineObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* isoline = (TimeAverageIsoline*)self->super.isoline_handle;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "Number of time average shall be an integer number of cycles");
            return -1;
        }
        int radius = (int)PyLong_AsLong(arg);

        try{
            isoline->setAverageCycles(radius);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyGetSetDef PyImanI_TimeAverageIsoline_Properties[] = {
            {(char*)"average_cycles", (getter)PyImanI_TimeAverageIsoline_GetAverageCycles,
                    (setter)PyImanI_TimeAverageIsoline_SetAverageCycles,
                    (char*)"time average radius, cycles\n"
                           "The isoline will be calculated by averaging the signal 'average_cycles' cycles before the \n"
                           "considering point and 'average_cycles' cycles after the considering point"},

            {NULL}
    };

    static int PyImanI_TimeAverageIsoline_Create(PyObject* module){

        PyImanI_TimeAverageIsolineType.tp_base = &PyImanI_IsolineType;
        PyImanI_TimeAverageIsolineType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanI_TimeAverageIsolineType.tp_doc =
                "The isoline will be calculated by smoothing the signal. This means that the isoline value at \n"
                "timestamp x is defined as signal average at the interval [x - N*C, x + N*C] where: \n"
                "C is a length of a single cycle in timestamps and N is some user-defined natural number\n";
        PyImanI_TimeAverageIsolineType.tp_init = (initproc)PyImanI_TimeAverageIsoline_Init;
        PyImanI_TimeAverageIsolineType.tp_getset = PyImanI_TimeAverageIsoline_Properties;

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
