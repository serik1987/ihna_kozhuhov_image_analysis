//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADER_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        void* trace_reader_handle;
        PyImanS_StreamFileTrainObject* file_train;
    } PyImanT_TraceReaderObject;

    static PyTypeObject PyImanT_TraceReaderType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.tracereader.TraceReader",
            .tp_basicsize = sizeof(PyImanT_TraceReaderObject),
            .tp_itemsize = 0,
    };

    static PyImanT_TraceReaderObject* PyImanT_TraceReader_New(PyTypeObject* type, PyObject* arg, PyObject* kwds){
        PyImanT_TraceReaderObject* self = NULL;
        self = (PyImanT_TraceReaderObject*)type->tp_alloc(type, 0);
        if (self != NULL){
            self->trace_reader_handle = NULL;
            self->file_train = NULL;
        }
        return self;
    }

    static int PyImanT_TraceReader_Init(PyImanT_TraceReaderObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;

        if (!PyArg_ParseTuple(args, "O!", &PyImanS_StreamFileTrainType, &self->file_train)){
            return -1;
        }

        Py_INCREF(self->file_train);

        auto* train = (StreamFileTrain*)self->file_train->super.train_handle;
        self->trace_reader_handle = new TraceReader(*train);

        return 0;
    }

    static void PyImanT_TraceReader_Destroy(PyImanT_TraceReaderObject* self){
        using namespace GLOBAL_NAMESPACE;

        if (self->trace_reader_handle != NULL){
            auto* reader = (TraceReader*)self->trace_reader_handle;
            delete reader;
        }

        Py_XDECREF(self->file_train);

        Py_TYPE(self)->tp_free(self);
    }

    static PyObject* PyImanT_TraceReader_GetArrivalTimeDisplacement(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        return PyLong_FromUnsignedLongLong(TraceReader::getArrivalTimeDisplacement());
    }

    static PyObject* PyImanT_TraceReader_GetSynchronizationChannelDisplacement(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        return PyLong_FromUnsignedLongLong(TraceReader::getSynchronizationChannelDisplacement());
    }

    static PyObject* PyImanT_TraceReader_GetFrameBodyDisplacement(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        return PyLong_FromUnsignedLongLong(reader->getFrameBodyDisplacement());
    }

    static PyObject* PyImanT_TraceReader_HasRead(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        try{
            return PyBool_FromLong(reader->hasRead());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetPixels(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        try{
            PyObject* pixel_list = PyList_New(0);
            if (pixel_list == NULL) return NULL;
            auto item = reader->getPixelItem(0);
            return pixel_list;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetTraces(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        try{
            auto* traces = reader->getTraces();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetInitialFrame(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        try{
            return PyLong_FromLong(reader->getInitialFrame());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetFinalFrame(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        try{
            return PyLong_FromLong(reader->getFinalFrame());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetFrameNumber(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        try{
            return PyLong_FromLong(reader->getFrameNumber());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetValue(PyImanT_TraceReaderObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        int n;
        int idx;

        if (!PyArg_ParseTuple(args, "ii", &n, &idx)){
            return NULL;
        }

        auto* reader = (TraceReader*)self->trace_reader_handle;

        try{
            double value = reader->getValue(n, idx);
            return PyFloat_FromDouble(value);
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetTrace(PyImanT_TraceReaderObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        int idx;
        auto* reader = (TraceReader*)self->trace_reader_handle;

        if (!PyArg_ParseTuple(args, "i", &idx)){
            return NULL;
        }

        try{
            if (idx < 0 || idx >= reader->getChannelNumber()){
                throw TraceReader::PixelItemIndexException();
            }
            const double* traces = reader->getTraces();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_Print(PyImanT_TraceReaderObject* self){
        using namespace std;
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;

        try{
            stringstream str;
            str << *reader;
            return PyUnicode_FromString(str.str().c_str());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyGetSetDef PyImanT_TraceReaderProperties[] = {
            {(char*)"arrival_time_displacement", (getter)PyImanT_TraceReader_GetArrivalTimeDisplacement, NULL,
             (char*)"arrival time displacement. This property is read-only becase is determined by \n"
             "the file format, not user"},

            {(char*)"synchronization_channel_displacement",
             (getter)PyImanT_TraceReader_GetSynchronizationChannelDisplacement,
             NULL,
             (char*)"displacement from the synchronization channel. This property is read-only, because \n"
                    "is determined by the file format, not user"},

            {(char*)"frame_body_displacement", (getter)PyImanT_TraceReader_GetFrameBodyDisplacement, NULL,
             (char*)"frame body displacement. This property is read-only, because is determined \n"
                    "by the file format, not user"},

            {(char*)"has_read", (getter)PyImanT_TraceReader_HasRead, NULL,
             (char*)"True if all traces have read by calling the read() method"},

            {(char*)"pixels", (getter)PyImanT_TraceReader_GetPixels, NULL,
             (char*)"coordinates of pixels which traces can be read by get_trace, get_value methods\n"
                    "or traces property\n"
                    "The returned result is a list. Element # i of the list corresponds to the trace # i.\n"
                    "Such an element is two-item tuple. In case if channel represents a signal from the map pixel\n"
                    "with index i on ordinate and index j on abscissa, the 0th element of the tuple corresponds to\n"
                    "abscissa and the 1st element is for the ordinate\n"
                    "In case if the channel # i is a synchronization channel, 0th element is a string 'SYNC' and \n"
                    "the 1st element is the synchronization channel number\n"
                    "In case if the channel # i is a time channel, 0th element is a string 'TIME' and \n"
                    "the 1st element is a time arrival\n"
                    "\n"
                    "This is a read-only property. Use add_pixels, remove_pixels, remove_all_pixels to change the \n"
                    "pixel list. Also, don't forget to read() the traces again\n"
                    "read()'ing the traces may change order of elements in this property as well as their number\n"},

            {(char*)"traces", (getter)PyImanT_TraceReader_GetTraces, NULL,
             (char*)"All traces as numpy array. Columns of the array correspond to traces while indices are timestamps\n"
                    "This is a read-only property because TraceReader is a trace reader, not writer\n"},

            {(char*)"initial_frame", (getter)PyImanT_TraceReader_GetInitialFrame, NULL,
             (char*)"Frame from which the reading starts\n"
                    "This is a read-only property. In order to set this property, use set_frame_range method"},

            {(char*)"final_frame", (getter)PyImanT_TraceReader_GetFinalFrame, NULL,
             (char*)"Frame at which the reading finishes\n"
                    "This is a read-only property. In order to set this property, use set_frame_range method"},

            {(char*)"frame_number", (getter)PyImanT_TraceReader_GetFrameNumber, NULL,
             (char*)"Total number of frames or timestamps in the read signal\n"
                    "This is a read-only property because this is fully defined by the initial_frame and final_frame"},

            {NULL}
    };

    static PyMethodDef PyImanT_TraceReaderMethods[] = {
            {"get_value", (PyCFunction)PyImanT_TraceReader_GetValue, METH_VARARGS,
             "Returns the value of the pixel for channel idx at timestamp n\n"
             "Please, note that read() will mix all pixel indices. Use pixels property t find an appropriate pixel index\n"
             "\n"
             "Usage: get_value(n, idx)\n"
             "Arguments:\n"
             "\tn - number of timestamp\n"
             "\tidx - channel number"},

            {"get_trace", (PyCFunction)PyImanT_TraceReader_GetTrace, METH_VARARGS,
             "Returns the signal from the trace with index idx\n"
             "Please, note that read() will mix all pixel indices. Use pixels property t find an appropriate pixel index\n"
             "\n"
             "Usage: get_trace(idx)\n"
             "Arguments:\n"
             "\tidx - pixel index"},

            {NULL}
    };

    int PyImanT_TraceReader_Create(PyObject* module){

        PyImanT_TraceReaderType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanT_TraceReaderType.tp_doc = "This is an engine to read the temporal-dependent signal from a certain \n"
                                         "map pixel, sychronization channel or arrival timestamp\n"
                                         "\n"
                                         "Usage: trace = TraceReader(train)\n"
                                         "where train is an instance of StreamFileTrain";
        PyImanT_TraceReaderType.tp_new = (newfunc)PyImanT_TraceReader_New;
        PyImanT_TraceReaderType.tp_init = (initproc)PyImanT_TraceReader_Init;
        PyImanT_TraceReaderType.tp_dealloc = (destructor)PyImanT_TraceReader_Destroy;
        PyImanT_TraceReaderType.tp_getset = PyImanT_TraceReaderProperties;
        PyImanT_TraceReaderType.tp_methods = PyImanT_TraceReaderMethods;
        PyImanT_TraceReaderType.tp_str = (reprfunc)PyImanT_TraceReader_Print;

        if (PyType_Ready(&PyImanT_TraceReaderType) < 0){
            return -1;
        }

        Py_INCREF(&PyImanT_TraceReaderType);

        if (PyModule_AddObject(module, "_tracereading_TraceReader", (PyObject*)&PyImanT_TraceReaderType) < 0){
            Py_DECREF(&PyImanT_TraceReaderType);
            return -1;
        }

        PyImanT_TraceReaderTypeHandle = (PyObject*)&PyImanT_TraceReaderType;

        return 0;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_TRACEREADER_H
