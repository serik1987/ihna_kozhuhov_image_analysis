//
// Created by serik1987 on 19.12.2019.
//

#include "Python.h"

extern "C" {

    static PyObject *PyExc_ImanError = NULL;
    static PyObject* PyExc_ImanIoError = NULL;
    static PyObject* PyExc_TrainError = NULL;
    static PyObject* PyExc_ExperimentModeError = NULL;
    static PyObject* PyExc_SynchronizationChannelNumberError = NULL;
    static PyObject* PyExc_UnsupportedExperimentModeError = NULL;
    static PyObject* PyExc_SourceFileError = NULL;

    static struct PyMethodDef core_methods[] = {
            {NULL}
    };

    static struct PyModuleDef core = {
            PyModuleDef_HEAD_INIT,
            .m_name = "_core",
            .m_doc = "for internal use only",
            .m_size = -1,
            .m_methods = core_methods
    };

    /**
     * Imports a single exception to the module
     *
     * @param module pointer to the module (created by PyModule_Create function)
     * @param exception_name class name corresponds to the exception
     * @param exception_full_name full exception name like packagename.modulename.ClassName
     * @param exception_doc short exception documentation
     * @param base_exception reference to the base exception class
     * @param exception_object reference to the reference to the exception object. When the exception
     * will be successfully created its reference will be written to the *exception_object
     * @param var_name name of an ultimate exception parameter that is not inherited by the base exception
     * or NULL if there is no such parameters
     * @return 0 on success, -1 on failure
     */
    static int import_exception(PyObject* module,
            const char* exception_name,
            const char* exception_full_name, const char* exception_doc,
            PyObject* base_exception,
            PyObject** exception_object,
            const char* var_name){

        PyObject* variables = NULL;
        PyObject* value = NULL;

        if (var_name != NULL){
            variables = PyDict_New();
            if (variables == NULL){
                printf("Trying to create the dictionary\n");
                return -1;
            }

            value = Py_BuildValue("");
            if (value == NULL){
                printf("Try to create the dictionary value\n");
                Py_DECREF(variables);
                return -1;
            }

            if (PyDict_SetItemString(variables, var_name, value) < 0){
                printf("Try to set the dictionary item\n");
                Py_DECREF(variables);
                return -1;
            }
        }

        *exception_object = PyErr_NewExceptionWithDoc(exception_full_name, exception_doc, base_exception, variables);
        if (*exception_object == NULL){
            Py_XDECREF(variables);
            Py_XDECREF(value);
            printf("Error in creating the following exception: %s\n", exception_name);
            return -1;
        }
        if (PyModule_AddObject(module, exception_name, *exception_object) < 0){
            Py_XDECREF(variables);
            Py_XDECREF(value);
            Py_DECREF(*exception_object);
            printf("Error in adding the following exception to the object: %s\n", exception_name);
            return -1;
        }

        return 0;
    }

    /**
     * Creates all exceptions necessary for ihna.kozhukhov.imageanalysis.sourcefiles module and adds them to
     * the module
     *
     * @param module reference to the module
     * @return 0 on success, -1 on failure
     */
    static int import_exceptions(PyObject* module){
        if (import_exception(module, "ImanError", "ihna.kozhukhov.imageanalysis.sourcefiles.ImanError",
                "The is the base class for all exceptions in the imageanalysis module", NULL, &PyExc_ImanError,
                NULL) < 0){
            return -1;
        }
        if (import_exception(module, "IoError", "ihna.kozhukhov.imageanalysis.sourcefiles.IoError",
                "This is the base class for all exceptions in the imageanalysis.sourcefiles module",
                PyExc_ImanError, &PyExc_ImanIoError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "TrainError", "ihna.kozhukhov.imageanalysis.sourcefiles.TrainError",
                "The is the base class for all I/O exceptions generated within the file train",
                PyExc_ImanIoError, &PyExc_TrainError, "train_name") < 0){
            return -1;
        }
        if (import_exception(module, "ExperimentModeError",
                "ihna.kozhukhov.imageanalysis.sourcefiles.ExperimentModeError",
                "This exception is thrown when you try to read the train property or call the train method that is"
                " absolutely meaningless for the current stimulation protocol. Also, when you try to call this method "
                "before the opening of the train, you will get this exception",
                PyExc_TrainError, &PyExc_ExperimentModeError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "SynchronizationChannelNumberError",
                "ihna.kozhukhov.iageanalysis.sourcefiles.SynchronizationChannelNumberError",
                "The exception is thrown when you try to access the non-existent synchronization channel or "
                "channel that was not used in the experiment",
                PyExc_TrainError, &PyExc_SynchronizationChannelNumberError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "UnsupportedExperimentModeError",
                "ihna.kozhukhov.imageanalysis.sourcefiles.UnsupportedExperimentModeError",
                "The experiment is thrown when image analysis have no idea or very confused about what stimulation"
                "protocol is used in your experiment. At the moment of creating an exception trying to read the"
                "file where both COST and EPST chunks were presented or where both of them are absent will"
                "throw this error",
                PyExc_TrainError, &PyExc_UnsupportedExperimentModeError, NULL) < 0){
            return -1;
        }
        if (import_exception(module, "SourceFileError",
                "ihna.kozhukhov.imageanalysis.sourcefiles.SourceFileError",
                "The is the base class for all exceptions that are connected to a certain particular file",
                PyExc_TrainError, &PyExc_SourceFileError, "file_name") < 0){
            return -1;
        }

        return 0;
    }

    static void clear_all_exceptions(void){
        Py_XDECREF(PyExc_ImanError);
        Py_XDECREF(PyExc_ImanIoError);
        Py_XDECREF(PyExc_TrainError);
        Py_XDECREF(PyExc_ExperimentModeError);
        Py_XDECREF(PyExc_SynchronizationChannelNumberError);
        Py_XDECREF(PyExc_SourceFileError);
    }

    PyMODINIT_FUNC PyInit__core(void){
        PyObject* module;

        module = PyModule_Create(&core);
        if (module == NULL){
            printf("Error in creating the module\n");
            return NULL;
        }
        if (import_exceptions(module) != 0){
            clear_all_exceptions();
            Py_DECREF(module);
            return NULL;
        }

        return module;
    }

}