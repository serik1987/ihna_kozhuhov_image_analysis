//
// Created by serik1987 on 18.02.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_MAPPLOTTER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_MAPPLOTTER_H

extern "C" {

    typedef struct {
        PyImanA_FrameAccumulatorObject parent;
    } PyImanA_MapPlotterObject;

    static PyTypeObject PyImanA_MapPlotterType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            /* tp_name */ "ihna.kozhukhov.imageanalysis.accumulators.MapPlotter",
            /* tp_basicsize */ sizeof(PyImanA_MapPlotterObject),
            /* tp_itemsize */ 0
    };

    static int PyImanA_MapPlotter_Init(PyImanA_MapPlotterObject* self, PyObject* args, PyObject* kwds){
        if (PyImanA_Accumulator_ArgumentCheck((PyImanA_AccumulatorObject*)self, args) < 0){
            return -1;
        }

        using namespace GLOBAL_NAMESPACE;
        try{
            auto* isoline_object = (PyImanI_IsolineObject*)self->parent.parent.corresponding_isoline;
            auto* isoline = (Isoline*)isoline_object->isoline_handle;
            auto* plotter = new MapPlotter(*isoline);
            self->parent.parent.handle = plotter;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return -1;
        }

        return 0;
    }

    static PyObject* PyImanA_MapPlotter_GetAveragedMap(PyImanA_MapPlotterObject* self, void*){
        using namespace GLOBAL_NAMESPACE;

        try{
            auto* plotter = (MapPlotter*)self->parent.parent.handle;
            auto& train = plotter->getTrain();
            const double* real_map = plotter->getRealMap();
            const double* img_map = plotter->getImaginaryMap();
            npy_intp dims[] = {train.getYSize(), train.getXSize()};
            PyObject* result = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
            for (unsigned int i=0; i < train.getYSize(); ++i){
                for (unsigned int j=0; j < train.getXSize(); ++j){
                    auto* pixel = (double*)PyArray_GETPTR2((PyArrayObject*)result, i, j);
                    *pixel = *(real_map++);
                    *(pixel+1) = *(img_map++);
                }
            }
            return result;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyGetSetDef PyImanA_MapPloter_Properties[] = {
            {(char*)"target_map", (getter)PyImanA_MapPlotter_GetAveragedMap, NULL,
                    (char*)"The resultant average map, or data processing results"},

            {NULL}
    };

    static int PyImanA_MapPlotter_Create(PyObject* module){

        PyImanA_MapPlotterType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanA_MapPlotterType.tp_base = &PyImanA_FrameAccumulatorType;
        PyImanA_MapPlotterType.tp_doc =
                "The object shall be used for plotting orientation, direction and amplitude maps\n"
                "\n"
                "Usage: plotter = MapPlotter(isoline)\n"
                "where isoline is ihna.kozhukhov.imageanalysis.isolines.Isoline object that is used for the isoline\n"
                "remove. This object also contains information about the synchronization and the train";
        PyImanA_MapPlotterType.tp_init = (initproc)PyImanA_MapPlotter_Init;
        PyImanA_MapPlotterType.tp_getset = PyImanA_MapPloter_Properties;

        if (PyType_Ready(&PyImanA_MapPlotterType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanA_MapPlotterType);
        PyImanA_MapPlotter_Handle = (PyObject*)&PyImanA_MapPlotterType;

        if (PyModule_AddObject(module, "_accumulators_MapPlotter", PyImanA_MapPlotter_Handle) < 0){
            return -1;
        }

        return 0;
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_MAPPLOTTER_H
