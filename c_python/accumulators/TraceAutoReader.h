//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEAUTOREADER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEAUTOREADER_H

extern "C" {

    typedef struct {
        PyImanA_AccumulatorObject parent;
    } PyImanA_TraceAutoReaderObject;

    static PyTypeObject PyImanA_TraceAutoReaderType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.accumulators.TraceAutoReader",
            /* tp_basicsize */ sizeof(PyImanA_TraceAutoReaderObject),
            /* tp_itemsize */ 0,
    };

    static int PyImanA_TraceAutoReader_Init(PyImanA_TraceAutoReaderObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        if (PyImanA_Accumulator_ArgumentCheck((PyImanA_AccumulatorObject*)self, args) < 0){
            return -1;
        }

        try{
            auto* isoline_object = (PyImanI_IsolineObject*)self->parent.corresponding_isoline;
            auto* isoline = (Isoline*)isoline_object->isoline_handle;
            auto* reader = new TraceAutoReader(*isoline);
            self->parent.handle = reader;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }

        return 0;
    }

    static int PyImanA_TraceAutoReader_Create(PyObject* module){

        PyImanA_TraceAutoReaderType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanA_TraceAutoReaderType.tp_base = &PyImanA_AccumulatorType;
        PyImanA_TraceAutoReaderType.tp_doc =
                "This accepts ROI and plots the trace averaged across ROI\n"
                "\n"
                "Usage: reader = TraceAutoReader(isoline)\n"
                "where isoline is an object that will remove isoline from your record\n"
                "(see ihna.kozhukhov.imageanalysis.isolines.Isoline for details)";
        PyImanA_TraceAutoReaderType.tp_init = (initproc)&PyImanA_TraceAutoReader_Init;

        if (PyType_Ready(&PyImanA_TraceAutoReaderType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanA_TraceAutoReaderType);
        PyImanA_TraceAutoReader_Handle = (PyObject*)&PyImanA_TraceAutoReaderType;

        if (PyModule_AddObject(module, "_accumulators_TraceAutoReader", PyImanA_TraceAutoReader_Handle) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEAUTOREADER_H
