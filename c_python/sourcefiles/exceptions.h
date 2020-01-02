//
// Created by serik1987 on 21.12.2019.
//

#ifndef IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
#define IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H

extern "C" {

    static PyObject* PyImanS_IoError = NULL;
    static PyObject* PyImanS_TrainError = NULL;
    static PyObject* PyImanS_TrainInitializationValue = NULL;
    static PyObject* PyImanS_TrainArguments = NULL;
    static PyObject* PyImanS_ExperimentModeError = NULL;
    static PyObject* PyImanS_SynchronizationChannelNumberError = NULL;
    static PyObject* PyImanS_UnsupportedExperimentModeError = NULL;
    static PyObject* PyImanS_SourceFileError = NULL;
    static PyObject* PyImanS_SourceFileInitializationValue = NULL;
    static PyObject* PyImanS_SourceFileErrorProperties = NULL;
    static PyObject* PyImanS_FrameNumberError = NULL;
    static PyObject* PyImanS_DataChunkError = NULL;
    static PyObject* PyImanS_IsoiChunkError = NULL;
    static PyObject* PyImanS_FileSizeError = NULL;
    static PyObject* PyImanS_ExperimentChunkError = NULL;
    static PyObject* PyImanS_FileHeaderError = NULL;
    static PyObject* PyImanS_FrameHeaderError = NULL;
    static PyObject* PyImanS_MapDimensionsError = NULL;
    static PyObject* PyImanS_DataTypeError = NULL;
    static PyObject* PyImanS_CompChunkNotFoundError = NULL;
    static PyObject* PyImanS_FileOpenError = NULL;
    static PyObject* PyImanS_FileReadError = NULL;
    static PyObject* PyImanS_ChunkError = NULL;
    static PyObject* PyImanS_ChunkError_initializer = NULL;
    static PyObject* PyImanS_ChunkError_argument = NULL;
    static PyObject* PyImanS_UnsupportedChunkError = NULL;
    static PyObject* PyImanS_ChunkSizeError = NULL;
    static PyObject* PyImanS_ChunkNotFoundError = NULL;
    static PyObject* PyImanS_FileNotOpenedError = NULL;
    static PyObject* PyImanS_IsoiChunkNotFoundError = NULL;
    static PyObject* PyImanS_FileNotLoadedError = NULL;
    static PyObject* PyImanS_DataChunkNotFoundError = NULL;
    static PyObject* PyImanS_NotAnalysisFileError = NULL;
    static PyObject* PyImanS_NotGreenFileError = NULL;
    static PyObject* PyImanS_NotInTrainHeadError = NULL;
    static PyObject* PyImanS_NotStreamFileError = NULL;
    static PyObject* PyImanS_NotCompressedFileError = NULL;
    static PyObject* PyImanS_FrameError = NULL;
    static PyObject* PyImanS_FrameError_init = NULL;
    static PyObject* PyImanS_FrameError_args = NULL;
    static PyObject* PyImanS_FrameNotReadError = NULL;
    static PyObject* PyImanS_FrameRangeError = NULL;
    static PyObject* PyImanS_FramChunkNotFoundError = NULL;
    static PyObject* PyImanS_CompressedFrameReadError = NULL;
    static PyObject* PyImanS_CacheSizeError = NULL;

    static int PyImanS_Create_exceptions(PyObject* module){
        PyImanS_IoError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.IoError",
                "This is the base class for all I/O errors", PyIman_ImanError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_IoError", PyImanS_IoError) < 0){
            return -1;
        }
        PyImanS_TrainInitializationValue = PyUnicode_FromString("[ === TRAIN INITIALIZATION VALUE === ]");
        PyImanS_TrainArguments = PyDict_New();
        if (PyDict_SetItemString(PyImanS_TrainArguments, "train_name", PyImanS_TrainInitializationValue) < 0){
            return -1;
        }
        PyImanS_TrainError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.TrainError",
                "This is the base class for all I/O errors occured during working with trains", PyImanS_IoError,
                PyImanS_TrainArguments);
        if (PyModule_AddObject(module, "_sourcefiles_TrainError", PyImanS_TrainError) < 0){
            return -1;
        }
        PyImanS_ExperimentModeError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.ExperimentModeError",
                "Experimental files and file trains may have properties or methods that are accessible only when the \n"
                "data were recorded under a certain stimulation protocol. Also, these properties/methods are not \n"
                "accessible until the file has not been read because the module have no idea about what stimulation \n"
                "protocol is applied. If you try to access such a data in the mentioned cases, you will receive this \n"
                "error",
                PyImanS_TrainError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_ExperimentModeError", PyImanS_ExperimentModeError) < 0){
            return -1;
        }
        PyImanS_SynchronizationChannelNumberError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.SynchronizationChannelNumberError",
                "The error will be thrown if you pass incorrect sychronization channel number as an argument",
                PyImanS_TrainError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_SynchronizationChannelNumberError",
                PyImanS_SynchronizationChannelNumberError) < 0){
            return -1;
        }
        PyImanS_UnsupportedExperimentModeError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.UnsupportedExperimentModeError",
                "This exception is thrown when this is not clear from the file header what stimulation protocol is \n"
                "used during the experiment. For example, when both COST and EPST chunks are presented or both of \n"
                "them are absent, this error will be definitely thrown",
                PyImanS_TrainError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_UnsupportedExperimentModeError",
                PyImanS_UnsupportedExperimentModeError) < 0){
            return -1;
        }
        PyImanS_SourceFileInitializationValue = PyUnicode_FromString("[=== SOURCE FILE INITIALIZATION VALUE ===]");
        PyImanS_SourceFileErrorProperties = PyDict_New();
        PyDict_SetItemString(PyImanS_SourceFileErrorProperties, "file_name", PyImanS_SourceFileInitializationValue);
        PyImanS_SourceFileError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.SourceFileError",
                "The is the base error for all I/O errors connected to a single data file",
                PyImanS_TrainError, PyImanS_SourceFileErrorProperties);
        if (PyModule_AddObject(module, "_sourcefiles_SourceFileError", PyImanS_SourceFileError) < 0){
            return -1;
        }
        PyImanS_FrameNumberError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.FrameNumberError",
                "In order to check the data consistency each file in the file train contains information about \n"
                "total number of frames in all record. During the file header read the module counts all frames found \n"
                "in all files belonging to this file train. This value is compared with total frame numbers read from "
                "the file header. If these values don't coincide to each other, this error will be thrown",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FrameNumberError", PyImanS_FrameNumberError) < 0){
            return -1;
        }
        PyImanS_DataChunkError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.DataChunkError",
                "If the data file is not corrupted the following condition is always valid:\n"
                "Size of the file body is the same as production of the frame number and size of the single frame\n"
                "If this condition doesn't hold, this error will be generated\n",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_DataChunkError", PyImanS_DataChunkError) < 0){
            return -1;
        }
        PyImanS_IsoiChunkError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.IsoiChunkError",
                "In order to check for file consistency the IMAN writes the file size to the file itself.\n"
                "File size shall be equal to ISOI chunk size plus ISOI chunk header size\n"
                "Such size shall be equal to size of the file header plus size of the file body\n"
                "if this is not true this error will be thrown\n",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_IsoiChunkError", PyImanS_IsoiChunkError) < 0){
            return -1;
        }
        PyImanS_FileSizeError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.FileSizeError",
                "In order to check for file consistency the IMAN writes the file size to the file itself.\n"
                "The file size shall be equal to ISOI chunk size plus ISOI chunk header size\n"
                "The sum of ISOI chunk and and ISOI chunk header size shall be equal to the actual number of file \n"
                "size revealed by the operating system. This error will be generated when this is not truth\n",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FileSizeError", PyImanS_FileSizeError) < 0){
            return -1;
        }
        PyImanS_ExperimentChunkError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.ExperimentChunkError",
                "The error is thrown when the stimulation protocol is not found in the current file or \n"
                "stimulation protocol is not the same for all files in the file train",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_ExperimentChunkError", PyImanS_ExperimentChunkError) < 0){
            return -1;
        }
        PyImanS_FileHeaderError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.FileHeaderError",
                "The error will be thrown when the file header is not the same for all files within the train",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FileHeaderError", PyImanS_FileHeaderError) < 0){
            return -1;
        }
        PyImanS_FrameHeaderError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.FrameHeaderError",
                "The error will be thrown when the frame header is not the same for all files within the train",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FrameHeaderError", PyImanS_FrameHeaderError) < 0){
            return -1;
        }
        PyImanS_MapDimensionsError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.MapDimensionsError",
                "The error will be thrown when the frame size on X or on Y is not the across all frames within \n"
                "the record",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_MapDimensionsError", PyImanS_MapDimensionsError) < 0){
            return -1;
        }
        PyImanS_DataTypeError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.DataTypeError",
                "The error occurs when the data type for a single map pixel is not the same across all record",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_DataTypeError", PyImanS_DataTypeError) < 0){
            return -1;
        }
        PyImanS_CompChunkNotFoundError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.CompChunkNotFoundError",
                "This error is thrown when you try to open the file train using CompressedFileTrain class but \n"
                "there is no evidence for the data to be compressed (i.e., no COMP chunk in the file header)\n",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_CompChunkNotFoundError", PyImanS_CompChunkNotFoundError) < 0){
            return -1;
        }
        PyImanS_FileOpenError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.FileOpenError",
                "This error is happend when the operating system is failed to open the file for reading",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FileOpenError", PyImanS_FileOpenError) < 0){
            return -1;
        }
        PyImanS_FileReadError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.FileReadError",
                "This error is happened when operating system is failed to read the file data or seek within the \n"
                "file position", PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FileReadError", PyImanS_FileReadError) < 0){
            return -1;
        }
        PyImanS_ChunkError_initializer = PyUnicode_FromString("[ ===== CHUNK ERROR: INITIALIZER ===== ]]");
        PyImanS_ChunkError_argument = PyDict_New();
        if (PyDict_SetItemString(PyImanS_ChunkError_argument, "chunk_name", PyImanS_ChunkError_initializer) < 0){
            return -1;
        }
        PyImanS_ChunkError = PyErr_NewExceptionWithDoc("ihna.kozhukhov.imageanalysis.sourcefiles.ChunkError",
                "The error will be thrown when the file contains chunk (a piece of the information) that can't be \n"
                "recognized by the current version of the module. In order to avoid this error use standard chunks \n"
                "only or update the module", PyImanS_SourceFileError, PyImanS_ChunkError_argument);
        if (PyModule_AddObject(module, "_sourcefiles_ChunkError", PyImanS_ChunkError) < 0){
            return -1;
        }
        PyImanS_UnsupportedChunkError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.UnsupportedChunkError",
                "The error will be thrown when the file contains chunk (a piece of the information) that can't be \n"
                "recognized by the current version of the module. In order to avoid this error use standard chunks \n"
                "only or update the module", PyImanS_ChunkError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_UnsupportedChunkError", PyImanS_UnsupportedChunkError) < 0){
            return -1;
        }
        PyImanS_ChunkSizeError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.ChunkSizeError",
                "The module assumes that size of all chunks except ISOI and DATA are highly fixed and absolutely \n"
                "predefined. Their sizes are written in the module library. The module routines always compare \n"
                "the desired chunk size written in the module library and the actual size of the chunk present in \n"
                "the reading file. If two values are not the same this error will be generated",
                PyImanS_ChunkError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_ChunkSizeError", PyImanS_ChunkSizeError) < 0){
            return -1;
        }
        PyImanS_ChunkNotFoundError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.ChunkNotFoundError",
                "The error is thrown when some mandatory chunk is absent in the reading file",
                PyImanS_ChunkError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_ChunkNotFoundError", PyImanS_ChunkNotFoundError) < 0){
            return -1;
        }
        PyImanS_FileNotOpenedError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.FileNotOpenedError",
                "This error will be thrown when you call the method that requires a file to be opened for reading",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FileNotOpenedError", PyImanS_FileNotOpenedError) < 0){
            return -1;
        }
        PyImanS_IsoiChunkNotFoundError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.IsoiChunkNotFoundError",
                "This error is thrown when there is not ISOI chunk presented at very beginning of the file",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_IsoiChunkNotFoundError", PyImanS_IsoiChunkNotFoundError) < 0){
            return -1;
        }
        PyImanS_FileNotLoadedError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.FileNotLoadedError",
                "This error is genereated when the calling method/property requires that file is opened for reading \n"
                "and its header is properly read", PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FileNotLoadedError", PyImanS_FileNotLoadedError) < 0){
            return -1;
        }
        PyImanS_DataChunkNotFoundError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.DataChunkNotFoundError",
                "This error is generated when you try to load the data with not DATA chunk",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_DataChunkNotFoundError", PyImanS_DataChunkNotFoundError) < 0){
            return -1;
        }
        PyImanS_NotAnalysisFileError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.NotAnalysisFileError",
                "This error will be generated when you try to use AnalysisSourceFile class to load the file that is \n"
                "not an IMAN analysis file",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_NotAnalysisFileError", PyImanS_NotAnalysisFileError) < 0){
            return -1;
        }
        PyImanS_NotGreenFileError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.NotGreenFileError",
                "This error will be generated when the file you try to open by means of GreeenSourceFile is not a \n"
                "green file",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_NotGreenFileError", PyImanS_NotGreenFileError) < 0){
            return -1;
        }
        PyImanS_NotInTrainHeadError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.NotInTrainHeadError",
                "The error will be thrown when you try to open stream file train or compressed file train or \n"
                "stream source file or traverse source file, the traverse mode is set to 'error' and file \n"
                "you pointed as an argument doesn't belong to the train head",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_NotInTrainHeadError", PyImanS_NotInTrainHeadError) < 0){
            return -1;
        }
        PyImanS_NotStreamFileError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.NotStreamFileError",
                "The error will be thrown when you try to read the file by means of StreamSourceFile or \n"
                "StreamFileError and file is far from being a stream file\n",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_NotStreamFileError", PyImanS_NotStreamFileError) < 0){
            return -1;
        }
        PyImanS_NotCompressedFileError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.NotCompressedFileError",
                "The error will be thrown when you try to read the file by means of CompressedSourceFile or \n"
                "CompressedFileTrain instances and the file doesn't contain the data in compressed state",
                PyImanS_SourceFileError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_NotCompressedFileError", PyImanS_NotCompressedFileError) < 0){
            return -1;
        }
        PyImanS_FrameError_init = PyLong_FromLong(-1);
        PyImanS_FrameError_args = PyDict_New();
        if (PyDict_SetItemString(PyImanS_FrameError_args, "frame_number", PyImanS_FrameError_init) < 0){
            return -1;
        }
        PyImanS_FrameError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.FrameError",
                "This is a general error happened during working with frames",
                PyImanS_TrainError, PyImanS_FrameError_args);
        if (PyModule_AddObject(module, "_sourcefiles_FrameError", PyImanS_FrameError) < 0){
            return -1;
        }
        PyImanS_FrameNotReadError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.FrameNotReadError",
                "This error is generated when you access the frame property that is required for a frame to be "
                "properly read from the file",
                PyImanS_FrameError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FrameNotReadError", PyImanS_FrameNotReadError) < 0){
            return -1;
        }
        PyImanS_FrameRangeError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.FrameRangeError",
                "This error is generated when you try to access the frame which number is not present in the train",
                PyImanS_FrameError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FrameRangeError", PyImanS_FrameRangeError) < 0){
            return -1;
        }
        PyImanS_FramChunkNotFoundError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.FramChunkNotFoundError",
                "Each frame shall be started from the FRAM chunk. If such chunk is not present in the frame this "
                "error will be generated",
                PyImanS_FrameError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_FramChunkNotFoundError", PyImanS_FramChunkNotFoundError) < 0){
            return -1;
        }
        PyImanS_CompressedFrameReadError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.CompressedFrameReadError",
                "All compressed files contain the frame #0 being uncompressed and all other frames being compressed. "
                "Hence, you can't read frames with number different from zero in the compressed files. "
                "Trying to do this will throw this exception",
                PyImanS_TrainError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_CompressedFrameReadError", PyImanS_CompressedFrameReadError) < 0){
            return -1;
        }
        PyImanS_CacheSizeError = PyErr_NewExceptionWithDoc(
                "ihna.kozhukhov.imageanalysis.sourcefiles.CacheSizeError",
                "If this exception is generated one of the following things were happened:\n"
                "1. Amount of free memory on your PC is not enough. You need to close some extra application or "
                "restart the Native data manager\n"
                "2. Cache capacity is not too little. Use train.capacity attribute to increase the cache capacity\n",
                PyImanS_TrainError, NULL);
        if (PyModule_AddObject(module, "_sourcefiles_CacheSizeError", PyImanS_CacheSizeError) < 0){
            return -1;
        }

        return 0;
    }

    static void PyImanS_Destroy_exceptions(){
        Py_XDECREF(PyImanS_IoError);
        Py_XDECREF(PyImanS_TrainError);
        Py_XDECREF(PyImanS_TrainInitializationValue);
        Py_XDECREF(PyImanS_TrainArguments);
        Py_XDECREF(PyImanS_ExperimentModeError);
        Py_XDECREF(PyImanS_SynchronizationChannelNumberError);
        Py_XDECREF(PyImanS_UnsupportedExperimentModeError);
        Py_XDECREF(PyImanS_SourceFileError);
        Py_XDECREF(PyImanS_SourceFileInitializationValue);
        Py_XDECREF(PyImanS_SourceFileErrorProperties);
        Py_XDECREF(PyImanS_FrameNumberError);
        Py_XDECREF(PyImanS_DataChunkError);
        Py_XDECREF(PyImanS_IsoiChunkError);
        Py_XDECREF(PyImanS_FileSizeError);
        Py_XDECREF(PyImanS_ExperimentChunkError);
        Py_XDECREF(PyImanS_FileHeaderError);
        Py_XDECREF(PyImanS_FrameHeaderError);
        Py_XDECREF(PyImanS_MapDimensionsError);
        Py_XDECREF(PyImanS_DataTypeError);
        Py_XDECREF(PyImanS_CompChunkNotFoundError);
        Py_XDECREF(PyImanS_FileOpenError);
        Py_XDECREF(PyImanS_FileReadError);
        Py_XDECREF(PyImanS_ChunkError);
        Py_XDECREF(PyImanS_ChunkError_initializer);
        Py_XDECREF(PyImanS_ChunkError_argument);
        Py_XDECREF(PyImanS_UnsupportedChunkError);
        Py_XDECREF(PyImanS_ChunkSizeError);
        Py_XDECREF(PyImanS_ChunkNotFoundError);
        Py_XDECREF(PyImanS_FileNotOpenedError);
        Py_XDECREF(PyImanS_IsoiChunkNotFoundError);
        Py_XDECREF(PyImanS_FileNotLoadedError);
        Py_XDECREF(PyImanS_DataChunkNotFoundError);
        Py_XDECREF(PyImanS_NotAnalysisFileError);
        Py_XDECREF(PyImanS_NotGreenFileError);
        Py_XDECREF(PyImanS_NotInTrainHeadError);
        Py_XDECREF(PyImanS_NotStreamFileError);
        Py_XDECREF(PyImanS_NotCompressedFileError);
        Py_XDECREF(PyImanS_FrameError);
        Py_XDECREF(PyImanS_FrameError_init);
        Py_XDECREF(PyImanS_FrameError_args);
        Py_XDECREF(PyImanS_FrameNotReadError);
        Py_XDECREF(PyImanS_FrameRangeError);
        Py_XDECREF(PyImanS_FramChunkNotFoundError);
        Py_XDECREF(PyImanS_CompressedFrameReadError);
        Py_XDECREF(PyImanS_CacheSizeError);
    }

    static int PyImanS_Exception_process(const void* handle){
        using namespace GLOBAL_NAMESPACE;

        int status = 0;
        auto* iman_handle = (iman_exception*)handle;
        auto* io_handle = dynamic_cast<io_exception*>(iman_handle);
        auto* data_chunk_read_error_handle = dynamic_cast<DataChunk::data_chunk_read_exception*>(iman_handle);

        if (data_chunk_read_error_handle != nullptr) {
            PyErr_SetString(PyExc_NotImplementedError, data_chunk_read_error_handle->what());
            status = -1;
        } else if (io_handle != nullptr){
            auto* train_handle = dynamic_cast<FileTrain::train_exception*>(io_handle);
            auto* source_file_handle = dynamic_cast<SourceFile::source_file_exception*>(io_handle);

            if (source_file_handle != nullptr) {
                PyObject* train_name = PyUnicode_FromString(source_file_handle->getTrainName().c_str());
                PyObject_SetAttrString(PyImanS_TrainError, "train_name", train_name);

                PyObject* file_name = PyUnicode_FromString(source_file_handle->getFilename().c_str());
                PyObject_SetAttrString(PyImanS_SourceFileError, "file_name", file_name);

                auto* frame_number_mismatch_handle =
                        dynamic_cast<FileTrain::frame_number_mismatch*>(source_file_handle);
                auto* data_chunk_mismatch_handle =
                        dynamic_cast<FileTrain::data_chunk_size_mismatch*>(source_file_handle);
                auto* isoi_chunk_mismatch_handle =
                        dynamic_cast<FileTrain::isoi_chunk_size_mismatch*>(source_file_handle);
                auto* file_size_mismatch_handle =
                        dynamic_cast<FileTrain::file_size_mismatch*>(source_file_handle);
                auto* experiment_chunk_not_found_handle =
                        dynamic_cast<FileTrain::experimental_chunk_not_found*>(source_file_handle);
                auto* file_header_mismatch_handle =
                        dynamic_cast<FileTrain::file_header_mismatch*>(source_file_handle);
                auto* frame_header_error_handle =
                        dynamic_cast<FileTrain::frame_header_mismatch*>(source_file_handle);
                auto* map_dimensions_error_handle =
                        dynamic_cast<FileTrain::map_dimensions_mismatch*>(source_file_handle);
                auto* data_type_error_handle =
                        dynamic_cast<FileTrain::data_type_mismatch*>(source_file_handle);
                auto* comp_chunk_not_found_handle =
                        dynamic_cast<CompressedFileTrain::comp_chunk_not_exist_exception*>(source_file_handle);
                auto* file_open_error_handle =
                        dynamic_cast<SourceFile::file_open_exception*>(source_file_handle);
                auto* file_read_error_handle =
                        dynamic_cast<SourceFile::file_read_exception*>(source_file_handle);
                auto* unsupported_chunk_error_handle =
                        dynamic_cast<SourceFile::unsupported_chunk_exception*>(source_file_handle);
                auto* chunk_size_error_handle =
                        dynamic_cast<SourceFile::chunk_size_mismatch_exception*>(source_file_handle);
                auto* chunk_not_found_error_handle =
                        dynamic_cast<SourceFile::chunk_not_found_exception*>(source_file_handle);
                auto* file_not_opened_error_handle =
                        dynamic_cast<SourceFile::file_not_opened*>(source_file_handle);
                auto* isoi_chunk_not_found_handle =
                        dynamic_cast<SourceFile::file_not_isoi_exception*>(source_file_handle);
                auto* file_not_loaded_handle =
                        dynamic_cast<SourceFile::file_not_loaded_exception*>(source_file_handle);
                auto* data_chunk_not_found_handle =
                        dynamic_cast<SourceFile::data_chunk_not_found_exception*>(source_file_handle);
                auto* not_analysis_file_handle =
                        dynamic_cast<AnalysisSourceFile::not_analysis_file_exception*>(source_file_handle);
                auto* not_green_file_handle =
                        dynamic_cast<GreenSourceFile::not_green_file_exception*>(source_file_handle);
                auto* not_in_train_head =
                        dynamic_cast<TrainSourceFile::not_train_head*>(source_file_handle);
                auto* not_stream_file_handle =
                        dynamic_cast<StreamSourceFile::not_stream_file*>(source_file_handle);
                auto* not_compressed_file_handle =
                        dynamic_cast<CompressedSourceFile::not_compressed_file_exception*>(source_file_handle);

                if (frame_number_mismatch_handle != nullptr) {
                    PyErr_SetString(PyImanS_FrameNumberError, frame_number_mismatch_handle->what());
                } else if (data_chunk_mismatch_handle != nullptr) {
                    PyErr_SetString(PyImanS_DataChunkError, data_chunk_mismatch_handle->what());
                } else if (isoi_chunk_mismatch_handle != nullptr) {
                    PyErr_SetString(PyImanS_IsoiChunkError, isoi_chunk_mismatch_handle->what());
                } else if (file_size_mismatch_handle != nullptr) {
                    PyErr_SetString(PyImanS_FileSizeError, file_size_mismatch_handle->what());
                } else if (experiment_chunk_not_found_handle != nullptr) {
                    PyErr_SetString(PyImanS_ExperimentChunkError, experiment_chunk_not_found_handle->what());
                } else if (file_header_mismatch_handle != nullptr) {
                    PyErr_SetString(PyImanS_FileHeaderError, file_header_mismatch_handle->what());
                } else if (frame_header_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_FrameHeaderError, frame_header_error_handle->what());
                } else if (map_dimensions_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_MapDimensionsError, map_dimensions_error_handle->what());
                } else if (data_type_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_DataTypeError, data_type_error_handle->what());
                } else if (comp_chunk_not_found_handle != nullptr) {
                    PyErr_SetString(PyImanS_CompChunkNotFoundError, comp_chunk_not_found_handle->what());
                } else if (file_open_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_FileOpenError, file_open_error_handle->what());
                } else if (file_read_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_FileReadError, file_read_error_handle->what());
                } else if (unsupported_chunk_error_handle != nullptr) {
                    PyObject *chunk_name = PyUnicode_FromString(unsupported_chunk_error_handle->getChunkName());
                    PyObject_SetAttrString(PyImanS_ChunkError, "chunk_name", chunk_name);
                    PyErr_SetString(PyImanS_UnsupportedChunkError, unsupported_chunk_error_handle->what());
                } else if (chunk_size_error_handle != nullptr) {
                    PyObject *chunk_name = PyUnicode_FromString(chunk_size_error_handle->what());
                    PyObject_SetAttrString(PyImanS_ChunkSizeError, "chunk_name", chunk_name);
                    PyErr_SetString(PyImanS_ChunkSizeError, chunk_size_error_handle->what());
                } else if (chunk_not_found_error_handle != nullptr) {
                    PyObject *chunk_name = PyUnicode_FromString(chunk_not_found_error_handle->getChunkName());
                    PyObject_SetAttrString(PyImanS_ChunkNotFoundError, "chunk_name", chunk_name);
                    PyErr_SetString(PyImanS_ChunkNotFoundError, chunk_not_found_error_handle->what());
                } else if (file_not_opened_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_FileNotOpenedError, file_not_opened_error_handle->what());
                } else if (isoi_chunk_not_found_handle != nullptr) {
                    PyErr_SetString(PyImanS_IsoiChunkNotFoundError, isoi_chunk_not_found_handle->what());
                } else if (file_not_loaded_handle != nullptr) {
                    PyErr_SetString(PyImanS_FileNotLoadedError, file_not_loaded_handle->what());
                } else if (data_chunk_not_found_handle != nullptr) {
                    PyErr_SetString(PyImanS_DataChunkNotFoundError, data_chunk_not_found_handle->what());
                } else if (not_analysis_file_handle != nullptr) {
                    PyErr_SetString(PyImanS_NotAnalysisFileError, not_analysis_file_handle->what());
                } else if (not_green_file_handle != nullptr) {
                    PyErr_SetString(PyImanS_NotGreenFileError, not_green_file_handle->what());
                } else if (not_in_train_head != nullptr) {
                    PyErr_SetString(PyImanS_NotInTrainHeadError, not_in_train_head->what());
                } else if (not_stream_file_handle != nullptr) {
                    PyErr_SetString(PyImanS_NotStreamFileError, not_stream_file_handle->what());
                } else if (not_compressed_file_handle != nullptr) {
                    PyErr_SetString(PyImanS_NotCompressedFileError, not_compressed_file_handle->what());
                } else {
                    PyErr_SetString(PyImanS_SourceFileError, source_file_handle->what());
                }

            } else if (train_handle != nullptr){
                PyObject* value = PyUnicode_FromString(train_handle->getFilename().c_str());
                PyObject_SetAttrString(PyImanS_TrainError, "train_name", value);

                auto* experiment_mode_handle = dynamic_cast<FileTrain::experiment_mode_exception*>(train_handle);
                auto* synchronization_channel_number_handle =
                        dynamic_cast<FileTrain::synchronization_channel_number_exception*>(train_handle);
                auto* unsupported_experiment_mode_error_handle =
                        dynamic_cast<FileTrain::unsupported_experiment_mode_exception*>(train_handle);
                auto* frame_error_handle = dynamic_cast<Frame::frame_exception*>(train_handle);
                auto* compressed_frame_read_handle =
                        dynamic_cast<CompressedFileTrain::compressed_frame_read_exception*>(train_handle);
                auto* cache_size_error_handle =
                        dynamic_cast<FileTrain::cache_too_small_exception*>(train_handle);

                if (frame_error_handle != nullptr) {
                    PyObject* frame_number = PyLong_FromLong(frame_error_handle->getFrameNumber());
                    PyObject_SetAttrString(PyImanS_FrameError, "frame_number", frame_number);

                    auto* frame_not_read_handle = dynamic_cast<Frame::frame_not_read*>(frame_error_handle);
                    auto* frame_range_error_handle = dynamic_cast<Frame::frame_is_out_of_range*>(frame_error_handle);
                    auto* fram_chunk_not_found_handle = dynamic_cast<Frame::fram_chunk_not_found_exception*>
                        (frame_error_handle);

                    if (frame_not_read_handle != nullptr) {
                        PyErr_SetString(PyImanS_FrameNotReadError, frame_not_read_handle->what());
                    } else if (frame_range_error_handle != nullptr) {
                        PyErr_SetString(PyImanS_FrameRangeError, frame_range_error_handle->what());
                    } else if (fram_chunk_not_found_handle != nullptr){
                        PyErr_SetString(PyImanS_FramChunkNotFoundError, fram_chunk_not_found_handle->what());
                    } else {
                        PyErr_SetString(PyImanS_FrameError, frame_error_handle->what());
                    }

                } else if (experiment_mode_handle != nullptr) {
                    PyErr_SetString(PyImanS_ExperimentModeError, experiment_mode_handle->what());
                } else if (synchronization_channel_number_handle != nullptr) {
                    PyErr_SetString(PyImanS_SynchronizationChannelNumberError,
                                    synchronization_channel_number_handle->what());
                } else if (unsupported_experiment_mode_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_UnsupportedExperimentModeError,
                                    unsupported_experiment_mode_error_handle->what());
                } else if (compressed_frame_read_handle != nullptr) {
                    PyErr_SetString(PyImanS_CompressedFrameReadError, compressed_frame_read_handle->what());
                } else if (cache_size_error_handle != nullptr) {
                    PyErr_SetString(PyImanS_CacheSizeError, cache_size_error_handle->what());
                } else {
                    PyErr_SetString(PyImanS_TrainError, train_handle->what());
                }
            } else {
                PyErr_SetString(PyImanS_IoError, io_handle->what());
            }
            status = -1;
        }

        return status;
    }

};

#endif //IHNA_KOZHUKHOV_IMAGE_ANALYSIS_EXCEPTIONS_H
