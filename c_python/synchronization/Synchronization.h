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
        printf("SO New Synchronization\n");
        auto* self = (PyImanY_SynchronizationObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->parent_train = NULL;
            self->synchronization_handle = NULL;
        }
        return self;
    }

    static void PyImanY_Synchronization_Destroy(PyImanY_SynchronizationObject* self){
        using namespace GLOBAL_NAMESPACE;

        printf("SO Destroy synchronization\n");
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
