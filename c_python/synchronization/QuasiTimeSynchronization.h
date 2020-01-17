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

    static PyObject* PyImanY_QuasiTimeSynchronization_GetStimulusPeriod
            (PyImanY_QuasiTimeSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiTimeSynchronization*)self->super.synchronization_handle;

        try{
            return PyFloat_FromDouble(sync->getStimulusPeriod());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_QuasiTimeSynchronization_SetStimulusPeriod
            (PyImanY_QuasiTimeSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiTimeSynchronization*)self->super.synchronization_handle;
        double period;

        if (!PyFloat_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "Stimulus period for the quasi-time synchronization must be integer");
            return -1;
        }
        period = PyFloat_AsDouble(arg);

        try{
            sync->setStimulusPeriod(period);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_QuasiTimeSynchronization_GetInitialCycle
            (PyImanY_QuasiTimeSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiTimeSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getInitialCycle());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_QuasiTimeSynchronization_SetInitialCycle
            (PyImanY_QuasiTimeSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiTimeSynchronization*)self->super.synchronization_handle;
        int init_cycle;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "Value for the initial cycle shall be integer");
            return -1;
        }
        init_cycle = (int)PyLong_AsLong(arg);

        try{
            sync->setInitialCycle(init_cycle);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_QuasiTimeSynchronization_GetFinalCycle
            (PyImanY_QuasiTimeSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiTimeSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getFinalCycle());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_QuasiTimeSynchronization_SetFinalCycle
            (PyImanY_QuasiTimeSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiTimeSynchronization*)self->super.synchronization_handle;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "final_cycle for QuasiTimeSynchronization shall be integer");
            return -1;
        }
        int final_cycle = (int)PyLong_AsLong(arg);

        try{
            sync->setFinalCycle(final_cycle);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyGetSetDef PyImanY_QuasiTimeSynchronization_Properties[] = {

            {(char*)"stimulus_period", (getter)PyImanY_QuasiTimeSynchronization_GetStimulusPeriod,
                    (setter)PyImanY_QuasiTimeSynchronization_SetStimulusPeriod,
                    (char*)"Stimulus period, ms"},

            {(char*)"initial_cycle", (getter)PyImanY_QuasiTimeSynchronization_GetInitialCycle,
             (setter)PyImanY_QuasiTimeSynchronization_SetInitialCycle,
             (char*)"Initial cycle\n"
                    "When the initial cycle is not set, this property equals to -1 before the synchronization\n"
                    "This means that this property will be set automatically during the synchronization in such\n"
                    "a way as to maximize the analysis epoch"},

            {(char*)"final_cycle", (getter)PyImanY_QuasiTimeSynchronization_GetFinalCycle,
             (setter)PyImanY_QuasiTimeSynchronization_SetFinalCycle,
             (char*)"Final cycle\n"
                    "When the final cycle is not set, this property equals to -1 before the synchronization\n"
                    "This means that this property will be set automatically during the synchronization in such\n"
                    "a way as to maximize the analysis epoch"},

            {NULL}
    };

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
                "defined by the user.\n"
                "\n"
                "Usage: sync = QuasiStimulusSynchronization(train)\n"
                "where: train is a certain train which is assumed to be opened";
        PyImanY_QuasiTimeSynchronizationType.tp_init = (initproc)PyImanY_QuasiTimeSynchronization_Init;
        PyImanY_QuasiTimeSynchronizationType.tp_getset = PyImanY_QuasiTimeSynchronization_Properties;

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
