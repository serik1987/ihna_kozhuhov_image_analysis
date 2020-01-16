//
// Created by serik1987 on 12.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SYNCHRONIZATION_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_SYNCHRONIZATION_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        PyImanS_StreamFileTrainObject* parent_train;
        void* synchronization_handle;
    } PyImanY_SynchronizationObject;

    static PyTypeObject PyImanY_SynchronizationType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.synchronization.Synchronization",
            .tp_basicsize = sizeof(PyImanY_SynchronizationObject),
    };

    static PyImanY_SynchronizationObject* PyImanY_Synchronization_New(PyTypeObject* type,
            PyObject* args, PyObject* kwds){
        auto* self = (PyImanY_SynchronizationObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->parent_train = NULL;
            self->synchronization_handle = NULL;
        }
        return self;
    }

    static void PyImanY_Synchronization_Destroy(PyImanY_SynchronizationObject* self){
        using namespace GLOBAL_NAMESPACE;

        Py_XDECREF(self->parent_train);

        if (self->synchronization_handle != NULL){
            auto* sync = (Synchronization*)self->synchronization_handle;
            delete sync;
        }

        Py_TYPE(self)->tp_free(self);
    }

    static int PyImanY_Synchronization_SetParent(PyImanY_SynchronizationObject* self, PyObject* args){

        PyObject* parent;

        if (!PyArg_ParseTuple(args, "O!", &PyImanS_StreamFileTrainType, &parent)){
            // What will happen if the type is not compatible
            return -1;
        }

        self->parent_train = (PyImanS_StreamFileTrainObject*)parent; // The type is definitely compatible!
                                                                     // Fuck you, CLion
        Py_INCREF(parent);

        return 0;
    }

    static int PyImanY_Synchronization_Init(PyImanY_SynchronizationObject* self, PyObject* args, PyObject* kwds){
        PyErr_SetString(PyExc_NotImplementedError, "Synchronization class is purely abstract. Don't create objects "
                                                   "of this class. Use any derived class for this purpose");
        return -1;
    }

    static PyObject* PyImanY_Synchronization_GetInitialFrame(PyImanY_SynchronizationObject* self, void*){

        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            return PyLong_FromLong(sync->getInitialFrame());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_Synchronization_GetFinalFrame(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            return PyLong_FromLong(sync->getFinalFrame());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_Synchronization_GetDoPrecise(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            return PyBool_FromLong(sync->isDoPrecise());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_Synchronization_SetDoPrecise(PyImanY_SynchronizationObject* self, PyObject* value, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;
        if (!PyBool_Check(value)){
            PyErr_SetString(PyExc_ValueError, "do_precise parameter requires boolean value");
            return -1;
        }

        try{
            sync->setDoPrecise(value == Py_True);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_Synchronization_GetSynchronizationPhase(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            const double* phase = sync->getSynchronizationPhase();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_Synchronization_GetPhaseIncrement(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            return PyFloat_FromDouble(sync->getPhaseIncrement());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_Synchronization_GetInitialPhase(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            return PyFloat_FromDouble(sync->getInitialPhase());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_Synchronization_GetReferenceCos(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            const double* ref = sync->getReferenceSignalCos();
            return Py_BuildValue("");
        } catch(std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_Synchronization_GetReferenceSin(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            const double* ref = sync->getReferenceSignalSin();
            return Py_BuildValue("");
        } catch(std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanY_Synchronization_GetHarmonic(PyImanY_SynchronizationObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            return PyFloat_FromDouble(sync->getHarmonic());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static int PyImanY_Synchronization_SetHarmonic(PyImanY_SynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;
        if (!PyFloat_Check(arg)){
            PyErr_SetString(PyExc_ValueError, "Bad value for harmonic property");
            return -1;
        }
        double value = PyFloat_AsDouble(arg);

        try{
            sync->setHarmonic(value);
            return 0;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }
    }

    static PyObject* PyImanY_Synchronization_GetSynchronized(PyImanY_SynchronizationObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* sync = (Synchronization*)self->synchronization_handle;

        try{
            return PyBool_FromLong(sync->isSynchronized());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyGetSetDef PyImanY_Synchronization_Properties[] = {
            {(char*)"initial_frame", (getter)PyImanY_Synchronization_GetInitialFrame, NULL,
                    (char*)"Initial frame from which the analysis starts\n"
                           "The is a read-only property. In order to set this property for NoSynchronization\n"
                           "please, use set_initial_frame() method\n"
                           "As for the other synchronization types, this parameter will be set automatically during\n"
                           "the synchronization process", NULL},

            {(char*)"final_frame", (getter)PyImanY_Synchronization_GetFinalFrame, NULL,
             (char*)"The frame where the analysis finishes\n"
                    "This is a read-only property. In order to set this property for NoSynchronization\n"
                    "please, use set_final_frame() method\n"
                    "As for the other synchronization types, this parameter will be set automatically during\n"
                    "the synchronization process", NULL},

            {(char*)"do_precise", (getter)PyImanY_Synchronization_GetDoPrecise,
            (setter)PyImanY_Synchronization_SetDoPrecise,
            (char*)"True if you need a precise synchronization, false otherwise\n"
                   "The precise synchronization requires more time but will given an exact result", NULL},

            {(char*)"synchronization_phase", (getter)PyImanY_Synchronization_GetSynchronizationPhase, NULL,
             (char*)"The synchronization phase as numpy array. The phase is dependent on the timestamp"},

            {(char*)"phase_increment", (getter)PyImanY_Synchronization_GetPhaseIncrement, NULL,
             (char*)"The averaged difference in stimulus phases between two consequtive frames\n"
                    "This is a read-only property because it will be set during the synchronization\n"
                    "When the train is not synchronize()'d this property equals to 0.0"},

            {(char*)"initial_phase", (getter)PyImanY_Synchronization_GetInitialPhase, NULL,
             (char*)"The stimulus phase for the very first phase subjected to the analysis\n"
                    "This ios a read-only property because it will be set during the synchronization\n"
                    "When the train is not synchronize()'d this property always equal to 0.0\n"
                    "This property is usually 0.0 if you the do_precise property is False"},

            {(char*)"reference_cos", (getter)PyImanY_Synchronization_GetReferenceCos, NULL,
             (char*)"Returns the reference cosine. Scalar-product the signal to this cosine to reveal the real part\n"
                    "of the averaged signal\n"
                    "This is a read-only property. It will be calculated during the synchronization"},

            {(char*)"reference_sin", (getter)PyImanY_Synchronization_GetReferenceSin, NULL,
             (char*)"The reference sine. Scalar-product the signal to this sine will reveal the imaginary part\n"
                    "of the averaged signal\n"
                    "This is a read-only property. It will be calculated during the synchronization"},

            {(char*)"harmonic", (getter)PyImanY_Synchronization_GetHarmonic,
             (setter)PyImanY_Synchronization_SetHarmonic,
             (char*)"Harmonic value. The harmonic is a ratio of the period of the stimulus value of interest to\n"
                    "the period of the stimulus selectivity. If the stimulus is drifting gratings using 2.0 \n"
                    "as harmonic will give an orientation selectivity maps and using 1.0 will give \n"
                    "direction selectivity maps. However, when the grating is stationary, 1.0 will give direction \n"
                    "selectivity"},

            {(char*)"synchronized", (getter)PyImanY_Synchronization_GetSynchronized, NULL,
             (char*)"True after successful synchronization process, False before this"},

            {NULL}

    };

    static int PyImanY_Synchronization_Create(PyObject* module){

        PyImanY_SynchronizationType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanY_SynchronizationType.tp_doc =
                "This is the base class for any synchronization object.\n"
                "The synchronization object contains interface and algorithms for providing the following steps:\n"
                "1) Reading / restoration of the synchronization signal\n"
                "Synchronization signal is a 1D matrix that shows how stimulus phase depends on the timestamp\n"
                "2) Setting the epoch for signal reading/analysis in such a way as it contains integer number \n"
                "of cycles\n"
                "3) Restoration of the reference sine and cosine. The resultant selectivity /preferred stimulus value\n"
                "is defined by the scalar production of the reference sine / cosine by the input signal without isoline\n"
                "\n"
                "The Synchronization class is abstract. You can't create objects from this class. Use any of its \n"
                "derived class. To reveal the list of all available derived classes please, dir() the Synchronization\n"
                "package";
        PyImanY_SynchronizationType.tp_new = (newfunc)PyImanY_Synchronization_New;
        PyImanY_SynchronizationType.tp_dealloc = (destructor)PyImanY_Synchronization_Destroy;
        PyImanY_SynchronizationType.tp_init = (initproc)PyImanY_Synchronization_Init;
        PyImanY_SynchronizationType.tp_getset = PyImanY_Synchronization_Properties;

        if (PyType_Ready(&PyImanY_SynchronizationType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanY_SynchronizationType);
        PyImanY_Synchronization_Handle = (PyObject*)&PyImanY_SynchronizationType;

        if (PyModule_AddObject(module, "_synchronization_Synchronization",
                (PyObject*)&PyImanY_SynchronizationType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_SYNCHRONIZATION_H
