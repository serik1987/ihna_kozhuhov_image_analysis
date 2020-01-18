//
// Created by serik1987 on 21.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H

#include <sstream>
#include "../cpp/exceptions.h"

extern "C" {

static PyObject *PyIman_ImanError = NULL;
static void PyIman_Exception_process(const void *);
static bool PyIman_ReadingProgressFunction(int processed, int total, const char* message, void* handle);

}

#include "sourcefiles/__init__.h"
#include "compression/__init__.h"
#include "tracereading/__init__.h"
#include "synchronization/__init__.h"
#include "isolines/__init__.h"

extern "C" {

    static bool PyIman_ReadingProgressFunction(int processed, int total, const char* message, void* handle){
        printf("SO %d out of %d steps were completed\n", processed, total);
        printf("SO Message: %s\n", message);
        return true;
    }

    static void PyIman_Exception_process(const void* handle){
        using namespace GLOBAL_NAMESPACE;

        auto* exception_handle = (std::exception*)handle;
        auto* iman_handle = dynamic_cast<iman_exception*>(exception_handle);

        if (iman_handle != NULL){
            if (PyImanS_Exception_process(handle) < 0) return;
            if (PyImanC_Exception_process(handle) < 0) return;
            if (PyImanT_Exception_process(handle) < 0) return;
            if (PyImanY_Exception_process(handle) < 0) return;
            if (PyImanI_Exception_Process(handle) < 0) return;
            PyErr_SetString(PyIman_ImanError, iman_handle->what());
        } else {
            PyErr_SetString(PyExc_RuntimeError, exception_handle->what());
        }
    }

}

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS___INIT___H
