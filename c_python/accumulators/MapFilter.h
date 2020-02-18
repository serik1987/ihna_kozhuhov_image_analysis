//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_MAPFILTER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_MAPFILTER_H

extern "C"{

    typedef struct {
        PyImanA_FrameAccumulatorObject parent;
    } PyImanA_MapFilterObject;

    static PyTypeObject PyImanA_MapFilterType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.accumulators.MapFilter",
            /* tp_basicsize */ sizeof(PyImanA_MapFilterObject),
            /* tp_itemsize */ 0
    };

    static int PyImanA_MapFilter_Init(PyImanA_MapFilterObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        if (PyImanA_Accumulator_ArgumentCheck((PyImanA_AccumulatorObject*)self, args) < 0){
            return -1;
        }

        try{
            auto* isoline_object = (PyImanI_IsolineObject*)self->parent.parent.corresponding_isoline;
            auto* isoline = (Isoline*)isoline_object->isoline_handle;
            auto* filter = new MapFilter(*isoline);
            self->parent.parent.handle = filter;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }

        return 0;
    }

    static int PyImanA_MapFilter_Create(PyObject* module){

        PyImanA_MapFilterType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanA_MapFilterType.tp_base = &PyImanA_FrameAccumulatorType;
        PyImanA_MapFilterType.tp_doc =
                "This object applies temporal filter to the imaging signal (w/o isoline) and then computes the signal\n"
                "squared-average across all frames.\n"
                "\n"
                "Usage: filter = MapFilter(isoline)\n"
                "where isoline is an object that removes an isoline (see ihna.kozhukhov.imageanalysis.isolines.Isoline\n"
                "for more details)";
        PyImanA_MapFilterType.tp_init = (initproc)PyImanA_MapFilter_Init;

        if (PyType_Ready(&PyImanA_MapFilterType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanA_MapFilterType);
        PyImanA_MapFilter_Handle = (PyObject*)&PyImanA_MapFilterType;

        if (PyModule_AddObject(module, "_accumulators_MapFilter", PyImanA_MapFilter_Handle) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPFILTER_H
