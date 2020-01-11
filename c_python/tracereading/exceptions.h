//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_tracereading_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_tracereading_EXCEPTIONS_H

extern "C" {

    static PyObject* PyImanT_TraceReaderError = NULL;
    static PyObject* PyImanT_PixelItemIndexError = NULL;
    static PyObject* PyImanT_TraceNotReadError = NULL;
    static PyObject* PyImanT_TimestampError = NULL;

    static int PyImanT_Exceptions_Create(PyObject* module){

        PyImanT_TraceReaderError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.tracereading.TraceReaderError",
                "This is the base class for all exceptions associated with a trace reading, except FileReadError",
                PyIman_ImanError, NULL);
        if (PyModule_AddObject(module, "_tracereading_TraceReaderError", PyImanT_TraceReaderError) < 0){
            return -1;
        }

        PyImanT_PixelItemIndexError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.tracereading.PixelItemIndexError",
                "This error will be called when pass index of not-existent trace",
                PyImanT_TraceReaderError, NULL);
        if (PyModule_AddObject(module, "_tracereading_PixelItemIndexError", PyImanT_PixelItemIndexError) < 0){
            return -1;
        }

        PyImanT_TraceNotReadError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.tracereading.TraceNotReadError",
                "This error will be thrown if you try to read the TraceReader property which is not available \n"
                "because the traces in the TraceReader have not been read()",
                PyImanT_TraceReaderError, NULL);
        if (PyModule_AddObject(module, "_tracereading_TraceNotReadError", PyImanT_TraceNotReadError) < 0){
            return -1;
        }

        PyImanT_TimestampError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.tracereading.TimestampError",
                "This error will be thrown if you try to get_value() from the trace and put the non-existent \n"
                "timestamp as a first argument",
                PyImanT_TraceReaderError, NULL);
        if (PyModule_AddObject(module, "_tracereading_TimestampError", PyImanT_TimestampError) < 0){
            return -1;
        }

        return 0;
    }

    static void PyImanT_Exceptions_Destroy(){
        Py_XDECREF(PyImanT_TraceReaderError);
        Py_XDECREF(PyImanT_PixelItemIndexError);
        Py_XDECREF(PyImanT_TraceNotReadError);
        Py_XDECREF(PyImanT_TimestampError);
    }

    static int PyImanT_Exception_process(const void* exception_handle){
        using namespace GLOBAL_NAMESPACE;
        auto* exception = (std::exception*)exception_handle;
        auto* trace_reader_handle = dynamic_cast<TraceReader::TraceReaderException*>(exception);

        if (trace_reader_handle != nullptr){
            auto* pixel_item_index_handle = dynamic_cast<TraceReader::PixelItemIndexException*>(exception);
            auto* trace_not_read_handle = dynamic_cast<TraceReader::TracesNotReadException*>(exception);
            auto* timestamp_handle = dynamic_cast<TraceReader::TimestampException*>(exception);

            if (pixel_item_index_handle != nullptr) {
                PyErr_SetString(PyImanT_PixelItemIndexError, pixel_item_index_handle->what());
            } else if (trace_not_read_handle != nullptr) {
                PyErr_SetString(PyImanT_TraceNotReadError, trace_not_read_handle->what());
            } else if (timestamp_handle != nullptr) {
                PyErr_SetString(PyImanT_TimestampError, timestamp_handle->what());
            } else {
                PyErr_SetString(PyImanT_TraceReaderError, trace_reader_handle->what());
            }
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
