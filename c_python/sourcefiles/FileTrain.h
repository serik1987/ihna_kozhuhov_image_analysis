//
// Created by serik1987 on 22.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_FILETRAIN_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_PY_FILETRAIN_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        void* train_handle;
    } PyImanS_FileTrainObject;

    static PyObject* PyImanS_FileTrain_New(PyTypeObject* type, PyObject*, PyObject*){
        PyImanS_FileTrainObject* self;
        self = (PyImanS_FileTrainObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->train_handle = NULL;
        }
        return (PyObject*)self;
    }

    static int PyImanS_FileTrain_Init(PyObject* self, PyObject* args, PyObject* kwds){
        PyErr_SetString(PyExc_NotImplementedError, "FileTrain class is purely abstract. No objects can be created "
                                                   "from it");
        return -1;
    }

    static void PyImanS_FileTrain_Destroy(PyImanS_FileTrainObject* self){
        if (self->train_handle != NULL){
            delete (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
            self->train_handle = NULL;
        }
        Py_TYPE(self)->tp_free((PyObject*)self);
    }

    static PyObject* PyImanS_FileTrain_GetFileNumber(PyImanS_FileTrainObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        return PyLong_FromSize_t(ptrain->getFileNumber());
    }

    static PyObject* PyImanS_FileTrain_GetFilePath(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyUnicode_FromString(ptrain->getFilePath().c_str());
    }

    static PyObject* PyImanS_FileTrain_GetFileName(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyUnicode_FromString(ptrain->getFilename().c_str());
    }

    static PyObject* PyImanS_FileTrain_GetFrameHeaderSize(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyLong_FromUnsignedLong(ptrain->getFrameHeaderSize());
    }

    static PyObject* PyImanS_FileTrain_GetFileHeaderSize(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyLong_FromUnsignedLong(ptrain->getFileHeaderSize());
    }

    static PyObject* PyImanS_FileTrain_IsOpened(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyBool_FromLong(ptrain->isOpened());
    }

    static PyObject* PyImanS_FileTrain_GetExperimentMode(PyImanS_FileTrainObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        FileTrain::ExperimentalMode emode = ptrain->getExperimentalMode();
        if (emode == FileTrain::Continuous){
            return PyUnicode_FromString("continuous");
        } else if (emode == FileTrain::Episodic){
            return PyUnicode_FromString("episodic");
        } else {
            return PyUnicode_FromString("unknown");
        }
    }

    static PyObject* PyImanS_FileTrain_GetFrameShape(PyImanS_FileTrainObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        uint32_t x = ptrain->getXSize();
        uint32_t y = ptrain->getYSize();
        return Py_BuildValue("kk", y, x);
    }

    static PyObject* PyImanS_FileTrain_GetFrameSize(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyLong_FromUnsignedLong(ptrain->getXYSize());
    }

    static PyObject* PyImanS_FileTrain_GetFrameImageSize(PyImanS_FileTrainObject* self, void*){
        auto ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyLong_FromSize_t(ptrain->getFrameImageSize());
    }

    static PyObject* PyImanS_FileTrain_GetTotalFrameSize(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyLong_FromSize_t(ptrain->getFrameSize());
    }

    static PyObject* PyImanS_FileTrain_GetSynchronizationChannelNumber(PyImanS_FileTrainObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        PyObject* result;

        try{
            result = PyLong_FromSize_t(ptrain->getSynchronizationChannelNumber());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            result = NULL;
        }

        return result;
    }

    static PyObject* PyImanS_FileTrain_GetSynchronizationChannelMax(PyImanS_FileTrainObject* self, PyObject* args,
            PyObject* keywords){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        int chan;
        PyObject* result;

        if (!PyArg_ParseTuple(args, "i", &chan)){
            return NULL;
        }

        try{
            result = PyLong_FromLong(ptrain->getSynchronizationChannelMax(chan));
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            result = NULL;
        }

        return result;
    }

    static PyObject* PyImanS_FileTrain_GetTotalFrames(PyImanS_FileTrainObject* self, void*){
        auto* ptrain = (GLOBAL_NAMESPACE::FileTrain*)self->train_handle;
        return PyLong_FromLong(ptrain->getTotalFrames());
    }

    static PyObject* PyImanS_FileTrain_Open(PyImanS_FileTrainObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        PyObject* status;

        try{
            ptrain->open();
            status = Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            status = NULL;
        }

        return status;
    }

    static PyObject* PyImanS_FileTrain_Close(PyImanS_FileTrainObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        PyObject* status;

        try{
            ptrain->close();
            status = Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            status = NULL;
        }

        return status;
    }

    static PyObject* PyImanS_FileTrain_Str(PyImanS_FileTrainObject* self){
        using namespace GLOBAL_NAMESPACE;
        auto* ptrain = (FileTrain*)self->train_handle;
        PyObject* status;

        try{
            std::stringstream ss;
            ss << *ptrain << std::endl;
            status = PyUnicode_FromString(ss.str().c_str());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            status = NULL;
        }

        return status;
    }

    static PyGetSetDef PyImanS_FileTrain_properties[] = {
            {(char*)"file_number", (getter)PyImanS_FileTrain_GetFileNumber, NULL,
                    (char*)"Total number of opened files in the train", NULL},
            {(char*)"file_path", (getter)PyImanS_FileTrain_GetFilePath, NULL,
                    (char*)"Full path to all files within the train", NULL},
            {(char*)"filename", (getter)PyImanS_FileTrain_GetFileName, NULL,
                    (char*)"Name of the head file of the train", NULL},
            {(char*)"frame_header_size", (getter)PyImanS_FileTrain_GetFrameHeaderSize, NULL,
                    (char*)"Size of the frame header in bytes or 4294967295 if file is not opened", NULL},
            {(char*)"file_header_size", (getter)PyImanS_FileTrain_GetFileHeaderSize, NULL,
                    (char*)"Size of the file header in bytes or 4294967295 if file is not opened", NULL},
            {(char*)"is_opened", (getter)PyImanS_FileTrain_IsOpened, NULL,
                    (char*)"Checks whether the train is opened for reading", NULL},
            {(char*)"experiment_mode", (getter)PyImanS_FileTrain_GetExperimentMode, NULL,
                    (char*)"Returns the description of the stimulation protocol", NULL},
            {(char*)"frame_shape", (getter)PyImanS_FileTrain_GetFrameShape, NULL,
                    (char*)"Returns the map dimensions on Y (1st tuple value) and on X (2nd tuple value)"},
            {(char*)"frame_size", (getter)PyImanS_FileTrain_GetFrameSize, NULL,
                    (char*)"Returns total number of pixels in the single frame", NULL},
            {(char*)"frame_image_size", (getter)PyImanS_FileTrain_GetFrameImageSize, NULL,
                    (char*)"Returns total size of the frame body, in bytes", NULL},
            {(char*)"total_frame_size", (getter)PyImanS_FileTrain_GetTotalFrameSize, NULL,
                    (char*)"Returns the total frame size in bytes (frame header size plus frame body size)", NULL},
            {(char*)"synchronization_channel_number", (getter)PyImanS_FileTrain_GetSynchronizationChannelNumber, NULL,
                    (char*)"Returns total number of synchronization channels (requires continuous stimulation protocol)", NULL},
            {(char*)"total_frames", (getter)PyImanS_FileTrain_GetTotalFrames, NULL,
                    (char*)"Returns total number of frames for the whole record", NULL},
            {NULL, NULL, NULL, NULL, NULL}
    };

    static PyMethodDef PyImanS_FileTrain_methods[] = {
            {"get_synchronization_channel_max", (PyCFunction)PyImanS_FileTrain_GetSynchronizationChannelMax,
             METH_VARARGS,
             "Returns the maximum value from a certain synchronization channel\n"
             "Arguments:\n"
             "\tchan - the channel number"},
            {"open", (PyCFunction)PyImanS_FileTrain_Open, METH_NOARGS, "Opens the file train for reading"},
            {"close", (PyCFunction)PyImanS_FileTrain_Close, METH_NOARGS, "Closes the file train"},
            {NULL, NULL, 0, NULL}
    };

    static PyTypeObject PyImanS_FileTrainType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.sourcefiles.FileTrain",
            .tp_basicsize = sizeof(PyImanS_FileTrainObject),
            .tp_itemsize = 0,
    };

    static int PyImanS_FileTrain_Create(PyObject* module){
        PyImanS_FileTrainType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
        PyImanS_FileTrainType.tp_doc =
                "This is the base class for all file trains\n"
                "\n"
                "In the imaging analysis the data are not stored within the single file but rather is splitted \n"
                "into several parts each of which is stored into a separate file. Hence, the data are \n"
                "represented in a sequence of separate files called 'file train'\n"
                "Example of the file train: T_1CK.0A00, T_1CK.0A01, T_1CK.0A02\n"
                "You can't read the data when you open a single file. Rather, you shall open the whole file train\n"
                "Use derivatives of this class in order to open the whole file train";
        PyImanS_FileTrainType.tp_new = PyImanS_FileTrain_New;
        PyImanS_FileTrainType.tp_dealloc = (destructor)PyImanS_FileTrain_Destroy;
        PyImanS_FileTrainType.tp_init = PyImanS_FileTrain_Init;
        PyImanS_FileTrainType.tp_getset = PyImanS_FileTrain_properties;
        PyImanS_FileTrainType.tp_methods = PyImanS_FileTrain_methods;
        PyImanS_FileTrainType.tp_str = (reprfunc)PyImanS_FileTrain_Str;

        if (PyType_Ready(&PyImanS_FileTrainType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanS_FileTrainType);

        if (PyModule_AddObject(module, "_sourcefiles_FileTrain", (PyObject*)&PyImanS_FileTrainType) < 0){
            return -1;
        }

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_FILETRAIN_H
