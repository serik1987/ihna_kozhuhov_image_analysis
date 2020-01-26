//
// Created by serik1987 on 08.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSION_PY_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_COMPRESSION_PY_EXCEPTIONS_H

#include "../../cpp/compression/Decompressor.h"

extern "C" {

static PyObject *PyImanC_DecompressionError = NULL;

static int PyImanC_Init_exceptions(PyObject *module) {

    PyImanC_DecompressionError = PyErr_NewExceptionWithDoc(
            "ihna.kozhukhov.imageanalysis.compression.DecompressionError",
            "This error is generated when the decompression and compression functions probably use different "
            "algorithms. This means that the data were compressed in older version of IMAN with no backward "
            "compatibility and shall be decompressed by this older version of IMAN",
            PyIman_ImanError, NULL);
    if (PyModule_AddObject(module, "_sourcefiles_DecompressionError", PyImanC_DecompressionError) < 0) {
        return -1;
    }

    return 0;
}

static void PyImanC_Destroy_exceptions() {
    Py_XDECREF(PyImanC_DecompressionError);
}

static int PyImanC_Exception_process(const void *exception_handle) {
    using namespace GLOBAL_NAMESPACE;
    auto *cpp_exception_handle = (std::exception *) exception_handle;
    auto *decompression_error_handle = dynamic_cast<Decompressor::decompression_exception *>(cpp_exception_handle);

    if (decompression_error_handle != nullptr) {
        PyErr_SetString(PyImanC_DecompressionError, decompression_error_handle->what());
        return -1;
    }

    return 0;
}

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
