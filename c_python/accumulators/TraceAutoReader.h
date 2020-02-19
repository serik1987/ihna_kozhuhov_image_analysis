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

    static PyObject* PyImanA_TraceAutoReader_SetRoi(PyImanA_TraceAutoReaderObject* self,
            PyObject* args, PyObject* kwds){

        using namespace GLOBAL_NAMESPACE;

        PyObject* roi;
        if (!PyArg_ParseTuple(args, "O", &roi)){
            return NULL;
        }

        PyObject* result = PyObject_CallMethod(roi, "get_coordinate_list", "");
        if (result == NULL){
            return NULL;
        }

        if (!PyList_Check(result)){
            Py_DECREF(result);
            PyErr_SetString(PyExc_ValueError, "The roi object shall have get_coordinate_list() method that shall "
                                              "return list of pixels available for the trace registration");
        }

        auto* reader = (TraceAutoReader*)self->parent.handle;

        try{
            reader->clearPixels();
            int pixel_number = PyList_Size(result);
            for (int i=0; i < pixel_number; ++i){
                PyObject* pixel_info = PyList_GetItem(result, i);
                if (!PyTuple_Check(pixel_info) && PyTuple_Size(pixel_info) != 2){
                    Py_DECREF(result);
                    PyErr_SetString(PyExc_ValueError, "Each item in the list returned by the ROI's get_coordinate_list "
                                                      "method shall be 2-item tuple");
                    return NULL;
                }
                PyObject* coordinate_object;
                int coordinates[2];
                for (int j=0; j < 2; ++j){
                    coordinate_object = PyTuple_GetItem(pixel_info, j);
                    if (!PyLong_Check(coordinate_object)){
                        Py_DECREF(result);
                        PyErr_SetString(PyExc_ValueError, "Pixel coordinate shall be integer\n");
                    }
                    coordinates[j] = PyLong_AsLong(coordinate_object);
                }
                PixelListItem pixel(*reader, coordinates[0], coordinates[1]);
                reader->addPixel(pixel);
            }
            PixelListItem time_channel(*reader, PixelListItem::ArrivalTime, 0);
            reader->addPixel(time_channel);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            Py_DECREF(result);
            return NULL;
        }

        Py_DECREF(result);
        printf("SO ROI was parsed successfully\n");
        return Py_BuildValue("");
    }

    static PyMethodDef PyImanA_TraceAutoReader_Methods[] = {
            {"set_roi", (PyCFunction)PyImanA_TraceAutoReader_SetRoi, METH_VARARGS,
             "Sets the ROI from which the signal will be averaged and saved\n"
             "Usage: set_roi(roi)\n"
             "where roi is usually an ihna.kozhukhov.imageanalysis.manifest.Roi instance\n"
             "(both SimpleRoi and ComplexRoi is applicable). However any object that have get_coordinate_list method\n"
             "with the same signature as Roi and the same format of output data is applicable"},

            {NULL}
    };

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
        PyImanA_TraceAutoReaderType.tp_methods = PyImanA_TraceAutoReader_Methods;

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
