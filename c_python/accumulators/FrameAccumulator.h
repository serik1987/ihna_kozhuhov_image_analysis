//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FRAMEACCUMULATOR_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_FRAMEACCUMULATOR_H

extern "C" {

    typedef struct {
        PyImanA_AccumulatorObject parent;
    } PyImanA_FrameAccumulatorObject;

    static PyTypeObject PyImanA_FrameAccumulatorType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.accumulators.FrameAccumulator",
            /* tp_basicsize */ sizeof(PyImanA_FrameAccumulatorObject),
            /* tp_itemsize */ 0,
    };

    static int PyImanA_FrameAccumulator_Create(PyObject* module){
        PyImanA_FrameAccumulatorType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanA_FrameAccumulatorType.tp_doc =
                "This is the base class for all frame accumulators\n"
                "The frame accumulator accepts the native imaging record, synchronizes it, removes all isolines \n"
                "from it and then computes the data averaged across all frames\n"
                "This is an abstract class. Don't use it. Use any of its derivatives";
        PyImanA_FrameAccumulatorType.tp_base = &PyImanA_AccumulatorType;

        if (PyType_Ready(&PyImanA_FrameAccumulatorType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanA_FrameAccumulatorType);
        PyImanA_FrameAccumulator_Handle = (PyObject*)&PyImanA_FrameAccumulatorType;

        if (PyModule_AddObject(module, "_accumulators_FrameAccumulator", PyImanA_FrameAccumulator_Handle) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FRAMEACCUMULATOR_H
