//
// Created by serik1987 on 11.01.2020.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADER_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_Py_TRACEREADER_H

extern "C" {

    typedef struct {
        PyObject_HEAD
        void *trace_reader_handle;
        PyImanS_StreamFileTrainObject *file_train;
        PyObject* progress_bar;
    } PyImanT_TraceReaderObject;

    static PyTypeObject PyImanT_TraceReaderType = {
            PyVarObject_HEAD_INIT(NULL, 0)
            .tp_name = "ihna.kozhukhov.imageanalysis.tracereader.TraceReader",
            .tp_basicsize = sizeof(PyImanT_TraceReaderObject),
            .tp_itemsize = 0,
    };

    typedef struct {
        PyObject_HEAD
        PyImanS_StreamFileTrainObject* parent_train;
        void* synchronization_handle;
    } PyImanY_SynchronizationObject;

    static PyObject* PyImanT_PixelListItem_AsTuple(PyImanT_TraceReaderObject* reader,
            const GLOBAL_NAMESPACE::PixelListItem& item);

    static PyImanT_TraceReaderObject *PyImanT_TraceReader_New(PyTypeObject *type, PyObject *arg, PyObject *kwds) {
        PyImanT_TraceReaderObject *self = NULL;
        self = (PyImanT_TraceReaderObject *) type->tp_alloc(type, 0);
        if (self != NULL) {
            self->trace_reader_handle = NULL;
            self->file_train = NULL;
            self->progress_bar = NULL;
        }
        return self;
    }

    static int PyImanT_TraceReader_Init(PyImanT_TraceReaderObject *self, PyObject *args, PyObject *kwds) {
        using namespace GLOBAL_NAMESPACE;

        if (!PyArg_ParseTuple(args, "O!", &PyImanS_StreamFileTrainType, &self->file_train)) {
            return -1;
        }

        Py_INCREF(self->file_train);
        Py_XDECREF(self->progress_bar);

        auto *train = (StreamFileTrain *) self->file_train->super.train_handle;
        self->trace_reader_handle = new TraceReader(*train);

        return 0;
    }

    static void PyImanT_TraceReader_Destroy(PyImanT_TraceReaderObject *self) {
        using namespace GLOBAL_NAMESPACE;

        if (self->trace_reader_handle != NULL) {
            auto *reader = (TraceReader *) self->trace_reader_handle;
            delete reader;
        }

        Py_XDECREF(self->file_train);
        Py_XDECREF(self->progress_bar);

        Py_TYPE(self)->tp_free(self);
    }

    static PyObject *PyImanT_TraceReader_GetArrivalTimeDisplacement(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        return PyLong_FromUnsignedLongLong(TraceReader::getArrivalTimeDisplacement());
    }

    static PyObject *PyImanT_TraceReader_GetSynchronizationChannelDisplacement(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        return PyLong_FromUnsignedLongLong(TraceReader::getSynchronizationChannelDisplacement());
    }

    static PyObject *PyImanT_TraceReader_GetFrameBodyDisplacement(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;
        return PyLong_FromUnsignedLongLong(reader->getFrameBodyDisplacement());
    }

    static PyObject *PyImanT_TraceReader_HasRead(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;
        try {
            return PyBool_FromLong(reader->hasRead());
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_GetPixels(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;
        try {
            PyObject *pixel_list = PyList_New(0);
            if (pixel_list == NULL) return NULL;
            for (int i=0; i < reader->getChannelNumber(); ++i){
                const PixelListItem& pixel = reader->getPixelItem(i);
                PyObject* pixel_object = PyImanT_PixelListItem_AsTuple(self, pixel);
                if (PyList_Append(pixel_list, pixel_object) < 0){
                    Py_DECREF(pixel_list);
                    Py_XDECREF(pixel_object);
                    return NULL;
                }
            }
            return pixel_list;
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_GetTraces(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;
        try {
            auto *traces = reader->getTraces();
            npy_intp dims[] = {reader->getFrameNumber(), reader->getChannelNumber()};
            PyObject* result = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
            if (result == NULL) return NULL;
            for (int i=0; i < reader->getFrameNumber(); ++i){
                for (int j=0; j < reader->getChannelNumber(); ++j, ++traces){
                    *(double*)PyArray_GETPTR2((PyArrayObject*)result, i, j) = *traces;
                }
            }
            return result;
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_GetInitialFrame(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;
        try {
            return PyLong_FromLong(reader->getInitialFrame());
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_GetFinalFrame(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;
        try {
            return PyLong_FromLong(reader->getFinalFrame());
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_GetFrameNumber(PyImanT_TraceReaderObject *self, void *) {
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;
        try {
            return PyLong_FromLong(reader->getFrameNumber());
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetChannelNumber(PyImanT_TraceReaderObject* self, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;

        try{
            return PyLong_FromLong(reader->getChannelNumber());
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_GetValue(PyImanT_TraceReaderObject *self, PyObject *args, PyObject *) {
        using namespace GLOBAL_NAMESPACE;
        int n;
        int idx;

        if (!PyArg_ParseTuple(args, "ii", &n, &idx)) {
            return NULL;
        }

        auto *reader = (TraceReader *) self->trace_reader_handle;

        try {
            double value = reader->getValue(n, idx);
            return PyFloat_FromDouble(value);
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_GetTrace(PyImanT_TraceReaderObject *self, PyObject *args, PyObject *) {
        using namespace GLOBAL_NAMESPACE;
        int idx;
        auto *reader = (TraceReader *) self->trace_reader_handle;

        if (!PyArg_ParseTuple(args, "i", &idx)) {
            return NULL;
        }

        try {
            if (idx < 0 || idx >= reader->getChannelNumber()) {
                throw TraceReader::PixelItemIndexException();
            }
            const double *traces = reader->getTraces() + idx;
            npy_intp dims[] = {reader->getFrameNumber()};
            auto* result = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
            if (result == NULL) return NULL;
            for (int i=0; i < reader->getFrameNumber(); ++i){
                *(double*)PyArray_GETPTR1((PyArrayObject*)result, i) = *traces;
                traces += reader->getChannelNumber();
            }
            return result;
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject *PyImanT_TraceReader_Print(PyImanT_TraceReaderObject *self) {
        using namespace std;
        using namespace GLOBAL_NAMESPACE;
        auto *reader = (TraceReader *) self->trace_reader_handle;

        try {
            stringstream str;
            str << *reader;
            return PyUnicode_FromString(str.str().c_str());
        } catch (std::exception &e) {
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_SetFrameRange(PyImanT_TraceReaderObject* self,
            PyObject* sync_arg, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        auto* sync_object = (PyImanY_SynchronizationObject*)sync_arg;
        auto* sync = (Synchronization*)sync_object->synchronization_handle;

        try{
            reader->setFrameRange(*sync);
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }
}

static GLOBAL_NAMESPACE::PixelListItem PyImanT_PixelListItem_FromTuple(PyImanT_TraceReaderObject* self,
        PyObject* tuple){
    using namespace GLOBAL_NAMESPACE;
    auto* reader = (TraceReader*)self->trace_reader_handle;

    if (!PyTuple_Check(tuple)){
        throw TraceReader::TraceNameException();
    }

    if (PyTuple_GET_SIZE(tuple) != 2){
        throw TraceReader::TraceNameException();
    }

    PyObject* first = PyTuple_GetItem(tuple, 0);
    PyObject* second = PyTuple_GetItem(tuple, 1);

    int i = -5, j;
    bool parsed;

    if (!PyLong_Check(second)){
        throw TraceReader::TraceNameException();
    }
    j = (int)PyLong_AsLong(second);

    if (PyUnicode_Check(first)){
        const char* chan_type = PyUnicode_AsUTF8(first);
        if (strcmp(chan_type, "TIME") == 0){
            i = PixelListItem::ARRIVAL_TIME;
            parsed = true;
        } else if (strcmp(chan_type, "SYNC") == 0){
            i = PixelListItem::SYNCH;
            parsed = true;
        } else {
            parsed = false;
        }
    } else if (PyLong_Check(first)){
        i = (int)PyLong_AsLong(first);
        parsed = true;
    } else {
        parsed = false;
    }

    if (parsed){
        return PixelListItem(*reader, i, j);
    } else {
        printf("i = %d, j = %d\n", i, j);
        throw TraceReader::TraceNameException();
    }
}

static PyObject* PyImanT_PixelListItem_AsTuple(PyImanT_TraceReaderObject* self,
        const GLOBAL_NAMESPACE::PixelListItem& item){
    using namespace GLOBAL_NAMESPACE;

    PyObject* tuple = PyTuple_New(2);
    if (tuple == NULL) return NULL;

    PyObject *first, *second;
    first = NULL;
    second = NULL;

    if (item.getPointType() == PixelListItem::ArrivalTime){
        first = PyUnicode_FromString("TIME");
        second = PyLong_FromLong(0);
    } else if (item.getPointType() == PixelListItem::SynchronizationChannel){
        first = PyUnicode_FromString("SYNC");
        second = PyLong_FromLong(item.getJ());
    } else if (item.getPointType() == PixelListItem::PixelValue){
        first = PyLong_FromLong(item.getI());
        second = PyLong_FromLong(item.getJ());
    } else {
        Py_DECREF(tuple);
        throw TraceReader::TraceNameException();
    }

    if (first == NULL || second == NULL){
        return NULL;
    }

    if (PyTuple_SetItem(tuple, 0, first) < 0 || PyTuple_SetItem(tuple, 1, second) < 0){
        Py_DECREF(first);
        Py_DECREF(second);
        Py_DECREF(tuple);
        return NULL;
    }

    return tuple;
}

extern "C"{

    static PyObject* PyImanT_TraceReader_PrintPixelValue(PyImanT_TraceReaderObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;

        try{
            PyObject* input;
            if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &input)){
                return NULL;
            }
            PixelListItem output = PyImanT_PixelListItem_FromTuple(self, input);
            std::cout << output;
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetArrivalTimePixel(PyImanT_TraceReaderObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;

        try{
            PixelListItem input(*reader, PixelListItem::ARRIVAL_TIME, 0);
            PyObject* output = PyImanT_PixelListItem_AsTuple(self, input);
            return output;
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetSynchronizationChannelPixel(PyImanT_TraceReaderObject* self,
            PyObject* args, PyObject*){

        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        int chan;

        if (!PyArg_ParseTuple(args, "i", &chan)){
            return NULL;
        }

        try{

            PixelListItem input(*reader, PixelListItem::SYNCH, chan);
            PyObject* output = PyImanT_PixelListItem_AsTuple(self, input);
            return output;

        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_GetPixelTrace(PyImanT_TraceReaderObject* self, PyObject* args, PyObject*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        int i, j;

        if (!PyArg_ParseTuple(args, "ii", &i, &j)){
            return NULL;
        }

        try{

            PixelListItem input(*reader, i, j);
            PyObject* output = PyImanT_PixelListItem_AsTuple(self, input);
            return output;

        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_PrintAllPixels(PyImanT_TraceReaderObject* self,
            PyObject* args, PyObject* kwds){

        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;

        try{
            reader->printAllPixels();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

    }

    static PyObject* PyImanT_TraceReader_AddPixel(PyImanT_TraceReaderObject* self,
            PyObject* args, PyObject* kwds){

        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        PyObject* arg;

        if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &arg)){
            return NULL;
        }

        try{
            PixelListItem pixel = PyImanT_PixelListItem_FromTuple(self, arg);
            reader->addPixel(pixel);
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

    }

    static PyObject* PyImanT_TraceReader_AddPixels(PyImanT_TraceReaderObject* self,
            PyObject* args, PyObject* kwds){

        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        PyObject* arg_list = NULL;

        if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &arg_list)){
            return NULL;
        }

        try{
            for (Py_ssize_t i=0; i < PyList_GET_SIZE(arg_list); ++i){
                PyObject* pixel_object = PyList_GetItem(arg_list, i);
                if (pixel_object == NULL) return NULL;
                PixelListItem item = PyImanT_PixelListItem_FromTuple(self, pixel_object);
                reader->addPixel(item);
            }
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }
    }

    static PyObject* PyImanT_TraceReader_ClearPixels(PyImanT_TraceReaderObject* self,
            PyObject* args, PyObject* kwds){

        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;

        try{
            reader->clearPixels();
            return Py_BuildValue("");
        } catch (std::exception& e){
            PyIman_Exception_process(&e);
            return NULL;
        }

    }

    static int PyImanT_TraceReader_SetProgressBar(PyImanT_TraceReaderObject* self, PyObject* arg, void*){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;
        int frame_number = reader->getFrameNumber();
        if (frame_number <= 0){
            frame_number = 10000;
        }

        PyObject* result = PyObject_CallMethod(arg, "progress_function", "iis", 0, frame_number, "Trace reading");
        if (result == NULL){
            return -1;
        }
        Py_XDECREF(result);

        Py_XDECREF(self->progress_bar);
        Py_INCREF(arg);
        self->progress_bar = arg;
        reader->setProgressFunction(PyIman_ReadingProgressFunction, "Trace reading", self->progress_bar);

        return 0;
    }

    static PyObject* PyImanT_TraceReader_Read(PyImanT_TraceReaderObject* self, PyObject* args, PyObject* kwds){
        using namespace GLOBAL_NAMESPACE;
        auto* reader = (TraceReader*)self->trace_reader_handle;

        try{
            reader->read();
            return Py_BuildValue("");
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
                    "This means that if you read the trace by application of get_trace(i), exact coordinates\n"
                    "of this trace is given in the item with index i in the list\n"
                    "Such an element is two-item tuple. In case if channel represents a signal from the map pixel\n"
                    "with index i on ordinate and index j on abscissa, the 0th element of the tuple corresponds to\n"
                    "ordinate and the 1st element is for the abscissa\n"
                    "In case if the channel # i is a synchronization channel, 0th element is a string 'SYNC' and \n"
                    "the 1st element is the synchronization channel number\n"
                    "In case if the channel # i is a time channel, 0th element is a string 'TIME' and \n"
                    "the 1st element is a time arrival\n"
                    "\n"
                    "This is a read-only property. Use add_pixels, remove_pixels, remove_all_pixels to change the \n"
                    "pixel list. Also, don't forget to read() the traces again\n"
                    "read()'ing the traces may change order of elements in this property as well as their number\n"
                    "\n"
                    "In orde to print the detailed information about the pixel please, use print_pixel_info method"},

            {(char*)"traces", (getter)PyImanT_TraceReader_GetTraces, NULL,
             (char*)"All traces as numpy array. Columns of the array correspond to traces while indices are timestamps\n"
                    "This is a read-only property because TraceReader is a trace reader, not writer\n"
                    "the row # i in this 2D matrix contains all data reflected to the timestamp i\n"
                    "the column # j of this 2D matrix contains all data reflected to the trace channel j\n"
                    "to see what kind of data is stored in trace channel j use pixels[j] property"},

            {(char*)"initial_frame", (getter)PyImanT_TraceReader_GetInitialFrame, NULL,
             (char*)"Frame from which the reading starts\n"
                    "This is a read-only property. In order to set this property, use set_frame_range method"},

            {(char*)"final_frame", (getter)PyImanT_TraceReader_GetFinalFrame, NULL,
             (char*)"Frame at which the reading finishes\n"
                    "This is a read-only property. In order to set this property, use set_frame_range method"},

            {(char*)"frame_number", (getter)PyImanT_TraceReader_GetFrameNumber, NULL,
             (char*)"Total number of frames or timestamps in the read signal\n"
                    "This is a read-only property because it is fully defined by the initial_frame and final_frame"},

            {(char*)"channel_number", (getter)PyImanT_TraceReader_GetChannelNumber, NULL,
             (char*)"Total number of traces that will be read.\n"
                    "This is a read-only property. Use add_pixel, add_pixels and clear_pixels method to manipulate \n"
                    "the value of this property"},

            {(char*)"progress_bar", NULL, (setter)PyImanT_TraceReader_SetProgressBar,
             (char*)"Sets the progress bar that will be updated each time when the reader is in the progress\n"
                    "This is a white-only property\n"
                    "the property accepts any Python object that contains the following method:\n"
                    "def progress_function(self, processed_frames, total_frames, message)\n"
                    "\t...\n"
                    "\n"
                    "All its arguments are defined by the object. The method shall return True or False\n"
                    "If it returns False, the reading process will be interrupted"},

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
             "\tidx - pixel index. To see what kind of data will be returned use pixels[idx] property"},

            {"print_pixel_value", (PyCFunction)PyImanT_TraceReader_PrintPixelValue, METH_VARARGS,
             "Prints the information about a certain channel on the screen"
             "\n"
             "Usage: print_pixel_value(pix)\n"
             "Arguments:\n"
             "\tpix - channel information containing in the pixels property. For instance, in order to print \n"
             "information containing in the channel with index idx you can write the following:\n"
             "reader.print_pixel_value(reader.pixels[idx])"},

            {"get_arrival_time_pixel", (PyCFunction)PyImanT_TraceReader_GetArrivalTimePixel, METH_NOARGS,
             "Returns the designation of a channel where arrival times will be placed\n"
             "\n"
             "The function will return a 2-item tuple that shall be substituted to the add_pixel or add_pixels methods"},

            {"get_synchronization_channel_pixel", (PyCFunction)PyImanT_TraceReader_GetSynchronizationChannelPixel,
             METH_VARARGS,
             "Returns the designation of a synchronization channel with a certain number\n"
             "\n"
             "Usage: get_synchronization_channel_pixel(chan)\n"
             "Arguments:\n"
             "\tchan - synchronization channel number\n"
             "The function will return a two-element tuple that shall be substituted into add_pixel and add_pixels\n"
             "method if you want a certain synchronization channel to be read"},

            {"get_pixel_trace", (PyCFunction)PyImanT_TraceReader_GetPixelTrace, METH_VARARGS,
             "Returns the designation of a trace channel that contains data from a stand-alone pixel on the map\n"
             "\n"
             "Usage: get_pixel_trace(i, j)\n"
             "Arguments:\n"
             "\ti - ordinate of this pixel\n"
             "\tj - abscissa of this pixel"},

            {"print_all_pixels", (PyCFunction)PyImanT_TraceReader_PrintAllPixels, METH_NOARGS,
             "Prints information about all pixels.\n"
             "The function doesn't require any arguments"},

            {"add_pixel", (PyCFunction)PyImanT_TraceReader_AddPixel, METH_VARARGS,
             "Adds a pixel to read from\n"
             "\n"
             "Usage: add_pixel(coordinates)\n"
             "The only argument depends on what kind of pixel you would like to read:\n"
             "('TIME', 0) is a pixel containing frame arrival times\n"
             "('SYNC', n) is a pixel containing data from synchronization channel n\n"
             "(i, j) is a pixel on the map with coordinates i on vertical and j on horizontal"},

            {"add_pixels", (PyCFunction)PyImanT_TraceReader_AddPixels, METH_VARARGS,
             "Adds several pixels to the pixel list\n"
             "\n"
             "Usage: add_pixels(coordinate_list)\n"
             "where coordinate_list is a list of tuples like those that shall be put into add_pixe argument"},

            {"clear_pixels", (PyCFunction)PyImanT_TraceReader_ClearPixels, METH_NOARGS,
             "Clears the pixel list\n"
             "The function requires no arguments"},

            {"read", (PyCFunction)PyImanT_TraceReader_Read, METH_NOARGS,
             "Reads all traces from the hard disk"},

            {"set_frame_range", (PyCFunction)PyImanT_TraceReader_SetFrameRange, METH_O,
             "Sets the initial and final frames of the analysis based on the synchronization results\n"
             "\n"
             "Usage: set_frame_range(sync)\n"
             "where sync is a Synchronization object (see ihna.kozhukhov.imageanalysis.synchronization.Synchronization)\n"
             "The sync shall be synchronize()'d"},

            {NULL}
    };

    int PyImanT_TraceReader_Create(PyObject* module){

        PyImanT_TraceReaderType.tp_flags = Py_TPFLAGS_BASETYPE | Py_TPFLAGS_DEFAULT;
        PyImanT_TraceReaderType.tp_doc = "This is an engine to read the temporal-dependent signal from a certain \n"
                                         "map pixel, synchronization channel or arrival timestamp\n"
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
