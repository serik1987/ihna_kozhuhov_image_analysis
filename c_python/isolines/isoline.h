//
// Created by serik1987 on 13.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ISOLINE_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_ISOLINE_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        void* parent_train;
        void* parent_synchronization;
        void* isoline_handle;
    } PyImanI_IsolineObject;

    static PyTypeObject PyImanI_IsolineType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.isolines.Isoline",
            .tp_basicsize = sizeof(PyImanI_IsolineObject),
            .tp_itemsize = 0,
    };

    static PyImanI_IsolineObject* PyImanI_Isoline_New(PyTypeObject* type, PyObject* args, PyObject* kwds){
        printf("SO New Isoline\n");
        auto* self = (PyImanI_IsolineObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->parent_train = NULL;
            self->parent_synchronization = NULL;
            self->isoline_handle = NULL;
        }
        return self;
    }

    static void PyImanI_Isoline_Destroy(PyImanI_IsolineObject* self){
        printf("SO Isoline destruction\n");
        using namespace GLOBAL_NAMESPACE;
        Py_XDECREF(self->parent_train);
        Py_XDECREF(self->parent_synchronization);
        if (self->isoline_handle != NULL){
            auto* isoline = (Isoline*)self->isoline_handle;
            delete isoline;
        }
        Py_TYPE(self)->tp_free(self);
    }

    static int PyImanI_Isoline_SetParent(PyImanI_IsolineObject* self, PyObject* args, PyObject* kwds){
        PyObject* train_object;
        PyObject* synchronization_object;

        if (!PyArg_ParseTuple(args, "O!O!", &PyImanS_StreamFileTrainType, &train_object,
                &PyImanY_SynchronizationType, &synchronization_object)){
            return -1;
        }

        Py_INCREF(train_object);
        Py_INCREF(synchronization_object);

        self->parent_train = train_object;
        self->parent_synchronization = synchronization_object;

        return 0;
    }

    static int PyImanI_Isoline_Init(PyImanI_IsolineObject* self, PyObject* args, PyObject* kwds){
        PyErr_SetString(PyExc_NotImplementedError,
                "the Isoline class is fully abstract. Use any of its derived classes");
        return -1;
    }

    static int PyImanI_Isoline_Create(PyObject* module){

        PyImanI_IsolineType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanI_IsolineType.tp_doc =
                "This class provides routines and parameters for isoline remove\n"
                "Isolines are ultra-slow changes in the intrinsic activity that occur at frequencies below the \n"
                "frequency of the stimulus. The isolines usually decline the accuracy of the analysis results because\n"
                "they require additional cycles for averaging."
                "In order to remove isolines:\n"
                "\t(a) create an instance of anly of its derivative class like\n"
                "\t\tisoline = TimeAverageIsoline(train, sync)"
                "\tAny isoline constructor requires two parameters. The first one is the source train. This train is \n"
                "\talways assumed to be opened. The second one is synchronization. Don't synchronize() the signal \n"
                "\tThe isoline() object will do it automatically when this is necessary\n"
                "\n"
                "\t(b) Set the isoline properties. The most of isoline object doesn't require to set any properties\n"
                "\tHowever, some of them like TimeAverageIsoline does require properties to be properly set\n"
                "\n"
                "\t(c) Use TraceReaderAndCleaner in order to reveal stand-alone traces without isolines\n"
                "\t(c1) Use scalar_product(...) function to get averaged maps where the isoline will be removed\n"
                "\n"
                "Use NoIsoline object if you don't want to remove the Isoline\n"
                "You can't create any instance of the Isoline class, you shall use any of its derived class instead";
        PyImanI_IsolineType.tp_new = (newfunc)PyImanI_Isoline_New;
        PyImanI_IsolineType.tp_dealloc = (destructor)PyImanI_Isoline_Destroy;
        PyImanI_IsolineType.tp_init = (initproc)PyImanI_Isoline_Init;

        if (PyType_Ready(&PyImanI_IsolineType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanI_IsolineType);
        PyImanI_Isoline_Handle = &PyImanI_IsolineType;

        if (PyModule_AddObject(module, "_isolines_Isoline", (PyObject*)&PyImanI_IsolineType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_ISOLINE_H
