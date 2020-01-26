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
            /* tp_name */ "ihna.kozhukhov.imageanalysis.synchronization.QuasiStimulusSynchronization",
            /* tp_basicsize */ sizeof(PyImanY_QuasiStimulusSynchronizationObject),
            /* tp_itemsize */ 0,
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
            auto* sync = new QuasiStimulusSynchronization(*train);
            self->super.synchronization_handle = sync;
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_QuasiStimulusSynchronization_GetStimulusPeriod
        (PyImanY_QuasiStimulusSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiStimulusSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getStimulusPeriod());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_QuasiStimulusSynchronization_SetStimulusPeriod
            (PyImanY_QuasiStimulusSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiStimulusSynchronization*)self->super.synchronization_handle;
        int period;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "Stimulus period shall be positive integer");
            return -1;
        }
        period = (int)PyLong_AsLong(arg);

        try{
            sync->setStimulusPeriod(period);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_QuasiStimulusSynchronization_GetInitialCycle
            (PyImanY_QuasiStimulusSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiStimulusSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getInitialCycle());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_QuasiStimulusSynchronization_SetInitialCycle
            (PyImanY_QuasiStimulusSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiStimulusSynchronization*)self->super.synchronization_handle;
        int value;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "initial cycle for the quasi-stimulus synchronization shall be integer");
            return -1;
        }
        value = (int)PyLong_AsLong(arg);

        try{
            sync->setInitialCycle(value);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_QuasiStimulusSynchronization_GetFinalCycle
            (PyImanY_QuasiStimulusSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiStimulusSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getFinalCycle());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_QuasiStimulusSynchronization_SetFinalCycle
            (PyImanY_QuasiStimulusSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (QuasiStimulusSynchronization*)self->super.synchronization_handle;
        int final_cycle;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "The final cycle for quasi-stimulus synchronization shall be integer");
            return -1;
        }
        final_cycle = (int)PyLong_AsLong(arg);

        try{
            sync->setFinalCycle(final_cycle);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyGetSetDef PyImanY_QuasiStimulusSynchronization_Properties[] = {
            {(char*)"stimulus_period", (getter)PyImanY_QuasiStimulusSynchronization_GetStimulusPeriod,
                    (setter)PyImanY_QuasiStimulusSynchronization_SetStimulusPeriod,
                    (char*)"Stimulus period, shall be an integer reflecting the whole number of frames"},

            {(char*)"initial_cycle", (getter)PyImanY_QuasiStimulusSynchronization_GetInitialCycle,
             (setter)PyImanY_QuasiStimulusSynchronization_SetInitialCycle,
             (char*)"The very first stimulus cycle included in the analysis\n"
                    "If you didn't set this value, it will be equal to -1 before synchronization\n"
                    "which means that the initial cycle \n"
                    "will be set automatically during the synchronization in such a way as to maximize the \n"
                    "analysis epoch"},

            {(char*)"final_cycle", (getter)PyImanY_QuasiStimulusSynchronization_GetFinalCycle,
             (setter)PyImanY_QuasiStimulusSynchronization_SetFinalCycle,
             (char*)"The very last stimulus cycle included in the analysis\n"
                    "If you didn't set the final cycle value this value will be equal to -1 before synchronization\n"
                    "This means that such value will be set automatically during the synchronization in such a way\n"
                    "as to maximize the analysis epoch"},

            {NULL}
    };

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
        PyImanY_QuasiStimulusSynchronizationType.tp_getset = PyImanY_QuasiStimulusSynchronization_Properties;

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

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_QUASISTIMULUSSYNCHRONIZATION_H
