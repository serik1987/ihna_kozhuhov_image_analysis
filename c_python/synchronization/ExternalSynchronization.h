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
            /* tp_name */ "ihna.kozhukhov.imageanalysis.synchronization.ExternalSynchronization",
            /* tp_basicsize */ sizeof(PyImanY_ExternalSynchronizationObject),
            /* tp_itemsize */ 0,
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

    static PyObject* PyImanY_ExternalSynchronization_GetChannelNumber
            (PyImanY_ExternalSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (ExternalSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getSynchronizationChannel());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_ExternalSynchronization_SetChannelNumber
            (PyImanY_ExternalSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (ExternalSynchronization*)self->super.synchronization_handle;

        if (!PyLong_Check(arg)){
            return -1;
        }
        int channel = (int)PyLong_AsLong(arg);

        try{
            sync->setSynchronizationChannel(channel);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_ExternalSynchronization_GetInitialCycle
            (PyImanY_ExternalSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (ExternalSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getInitialCycle());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_ExternalSynchronization_SetInitialCycle
            (PyImanY_ExternalSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (ExternalSynchronization*)self->super.synchronization_handle;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "initial cycle shall be integer");
            return -1;
        }
        int initial_cycle = (int)PyLong_AsLong(arg);

        try{
            sync->setInitialCycle(initial_cycle);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_ExternalSychronization_GetFinalCycle
            (PyImanY_ExternalSynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (ExternalSynchronization*)self->super.synchronization_handle;

        try{
            return PyLong_FromLong(sync->getFinalCycle());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_ExternalSynchronization_SetFinalCycle
            (PyImanY_ExternalSynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (ExternalSynchronization*)self->super.synchronization_handle;

        if (!PyLong_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "final cycle for the external synchronization shall be integer");
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

    static PyGetSetDef PyImanY_ExternalSynchronization_Properties[] = {

            {(char*)"channel_number", (getter)PyImanY_ExternalSynchronization_GetChannelNumber,
                    (setter)PyImanY_ExternalSynchronization_SetChannelNumber,
                    (char*)"channel that shall be used for the signal synchronization"},

            {(char*)"initial_cycle", (getter)PyImanY_ExternalSynchronization_GetInitialCycle,
             (setter)PyImanY_ExternalSynchronization_SetInitialCycle,
             (char*)"The very beginning cycle of the analysis\n"
                    "When the initial cycle is not set, this property equals to -1 before the synchronization\n"
                    "This means the the initial cycle will be set automatically during the synchronization\n"
                    "in such a way as to maximize the analysis epoch"},

            {(char*)"final_cycle", (getter)PyImanY_ExternalSychronization_GetFinalCycle,
             (setter)PyImanY_ExternalSynchronization_SetFinalCycle,
             (char*)"Cycle where the analysis finishes\n"
                    "When the final cycle is not set, this property equals to -1 before the synchronization\n"
                    "This means that the final cycle will be set automatically during the synchronization\n"
                    "in such a way as to maximize the analysis epoch"},

            {NULL}
    };

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
        PyImanY_ExternalSynchronizationType.tp_getset = PyImanY_ExternalSynchronization_Properties;

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

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXTERNALSYNCHRONIZATION_H
