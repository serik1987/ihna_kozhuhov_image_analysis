//
// Created by serik1987 on 19.12.2019.
//

#include "Python.h"
#include "exceptions.h"

extern "C" {

    static PyObject* c_api_test(PyObject* module, PyObject* args){
        /*
        if (PyImanS_Exception_process(NULL) < 0){
            return NULL;
        }
        return Py_BuildValue("");
         */
        // PyErr_SetString(PyImanS_ChunkSizeMismatchError, "The is a sample exception");
        PyObject* list = PyList_New(0);
        if (PyList_Append(list, Py_BuildValue("")) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ImanError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_IoError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_TrainError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ExperimentModeError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_SynchronizationChannelNumberError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_UnsupportedExperimentModeError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_SourceFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FrameNumberError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_DataChunkNotFoundError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_IsoiChunkSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ExperimentChunkMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileHeaderMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FrameHeaderMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FrameDimensionsMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_DataTypeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_CompChunkNotExistError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileNotOpenedError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileOpenError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileReadError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ChunkError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_UnsupportedChunkError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ChunkSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_ChunkNotFoundError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_IsoiChunkNotFoundError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_FileNotLoadedError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_DataChunkSizeMismatchError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotAnalysisFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotGreenFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotCompressedFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotStreamFileError) < 0) return NULL;
        if (PyList_Append(list, PyImanS_NotTrainHeadError) < 0) return NULL;
        return list;
    }

    static struct PyMethodDef test_methods[] = {
            {"c_api_test", c_api_test, METH_NOARGS, "This is a test function"},
            {NULL}
    };

    static struct PyModuleDef test_module = {
            PyModuleDef_HEAD_INIT,
            .m_name = "test",
            .m_doc = "this is a test module",
            .m_size = -1,
            .m_methods = test_methods
    };

    PyMODINIT_FUNC PyInit_test(void){
        PyObject* module = NULL;

        module = PyModule_Create(&test_module);
        if (module == NULL) return NULL;

        if (PyImanS_Import__exceptions() < 0){
            Py_DECREF(module);
            return NULL;
        }

        return module;
    }

}